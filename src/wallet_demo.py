import argparse
import json
import secrets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from sklearn.metrics import roc_curve

from eeg_channels import CHANNEL_NAMES
from eeg_embedding_model import EEGEmbeddingNet
from paths import CACHE_ROOT, resolve_cache_file

NUM_SUBJECTS = 100
TEST_FREQUENCIES = (8.5, 9.5, 10.5, 11.5, 12.0)
ENROLLMENT_SESSION = 4
CALIBRATION_SESSION = 5
TEST_SESSION = 6
WEB_ROOT = Path(__file__).resolve().parent.parent / "wallet-demo"


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-12)


class DemoAuthService:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.channel_indices = tuple(
            self.checkpoint.get("channel_indices", ())
        )
        if len(self.channel_indices) != 8:
            raise ValueError(
                "The wallet demo requires the trained 8-channel checkpoint."
            )
        if any(
            index < 0 or index >= len(CHANNEL_NAMES)
            for index in self.channel_indices
        ):
            raise ValueError("Model checkpoint has invalid channel indices.")

        self.training_frequencies = tuple(
            float(frequency)
            for frequency in self.checkpoint["training_frequencies"]
        )
        self.training_sessions = tuple(
            int(session)
            for session in self.checkpoint["training_sessions"]
        )
        if set(self.training_frequencies).intersection(TEST_FREQUENCIES):
            raise ValueError(
                "Wallet demo test frequencies must be excluded from training."
            )

        self.model = EEGEmbeddingNet(
            num_subjects=int(self.checkpoint["num_subjects"]),
            embedding_size=int(self.checkpoint["embedding_size"]),
            num_channels=len(self.channel_indices),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        self.metadata = []
        for metadata_name in ("metadata.json", "test_metadata.json"):
            metadata_path = CACHE_ROOT / metadata_name
            if metadata_path.is_file():
                with open(metadata_path, "r", encoding="utf-8") as file:
                    self.metadata.extend(json.load(file))
        if not self.metadata:
            raise FileNotFoundError(
                f"No cache metadata found under {CACHE_ROOT}."
            )

        enrollment_items = self._select_items(
            ENROLLMENT_SESSION,
            self.training_frequencies,
        )
        self.templates = self._build_templates(enrollment_items)
        calibration_items = self._select_items(
            CALIBRATION_SESSION,
            TEST_FREQUENCIES,
        )
        calibration_embeddings, calibration_subjects = self._extract(
            calibration_items
        )
        calibration_scores, calibration_labels = self._score_all(
            calibration_embeddings,
            calibration_subjects,
        )
        fpr, tpr, thresholds = roc_curve(
            calibration_labels,
            calibration_scores,
        )
        fnr = 1.0 - tpr
        eer_index = int(np.argmin(np.abs(fpr - fnr)))
        self.threshold = float(thresholds[eer_index])
        self.calibration_eer = float((fpr[eer_index] + fnr[eer_index]) / 2)
        self.trials = {
            (int(item["subject"]), float(item["frequency"])): item
            for item in self._select_items(TEST_SESSION, TEST_FREQUENCIES)
        }
        missing_trials = [
            (subject, frequency)
            for subject in range(1, NUM_SUBJECTS + 1)
            for frequency in TEST_FREQUENCIES
            if (subject, frequency) not in self.trials
        ]
        if missing_trials:
            raise ValueError(
                "Test cache is incomplete; missing "
                f"{len(missing_trials)} subject/frequency trials."
            )

        print("Wallet demo model device:", self.device)
        print("Demo profiles:", NUM_SUBJECTS, "(anonymized dataset IDs)")
        print("Model channels:", self.checkpoint.get("channel_names", []))
        print("Calibration threshold:", f"{self.threshold:.6f}")
        print(
            "Calibration EER:",
            f"{self.calibration_eer:.6f} ({self.calibration_eer * 100:.4f}%)",
        )
        print(
            "WARNING: authentication uses public, replayable dataset trials; "
            "this is not real biometric security."
        )

    def _select_items(self, session, frequencies):
        items = [
            item
            for item in self.metadata
            if int(item["session"]) == session
            and float(item["frequency"]) in frequencies
        ]
        subjects = {int(item["subject"]) for item in items}
        if subjects != set(range(1, NUM_SUBJECTS + 1)):
            raise ValueError(
                f"Session {session} must contain trials for all "
                f"{NUM_SUBJECTS} subjects; found {len(subjects)}."
            )
        if not items:
            raise ValueError(
                f"No cached trials for session {session} and frequencies "
                f"{tuple(frequencies)}."
            )
        return items

    def _extract(self, items):
        eeg_trials = []
        subjects = []
        for item in items:
            eeg = np.load(resolve_cache_file(item["file"])).astype(np.float32)
            if eeg.shape != (len(CHANNEL_NAMES), 1500):
                raise ValueError(
                    f"Expected cached trial shape "
                    f"({len(CHANNEL_NAMES)}, 1500), got {eeg.shape}."
                )
            eeg_trials.append(eeg[list(self.channel_indices), :])
            subjects.append(int(item["subject"]))

        embeddings = []
        with torch.no_grad():
            for start in range(0, len(eeg_trials), 32):
                batch = torch.from_numpy(
                    np.asarray(eeg_trials[start : start + 32])[:, None, :, :]
                ).to(self.device)
                batch_embeddings, _ = self.model(batch)
                embeddings.append(batch_embeddings.cpu().numpy())

        return (
            normalize_embeddings(np.concatenate(embeddings, axis=0)),
            np.asarray(subjects, dtype=int),
        )

    def _build_templates(self, items):
        embeddings, subjects = self._extract(items)
        templates = {}
        for subject in range(1, NUM_SUBJECTS + 1):
            subject_embeddings = embeddings[subjects == subject]
            if len(subject_embeddings) == 0:
                raise ValueError(f"Missing enrollment data for subject {subject}.")
            template = subject_embeddings.mean(axis=0)
            templates[subject] = template / (
                np.linalg.norm(template) + 1e-12
            )
        return templates

    def _score_all(self, embeddings, subjects):
        scores = []
        labels = []
        for embedding, true_subject in zip(embeddings, subjects):
            for subject, template in self.templates.items():
                scores.append(float(np.dot(embedding, template)))
                labels.append(int(subject == int(true_subject)))
        return np.asarray(scores), np.asarray(labels)

    def authenticate(self, profile_subject, sample_subject, frequency):
        if not 1 <= profile_subject <= NUM_SUBJECTS:
            raise ValueError("Profile ID must be between 1 and 100.")
        if not 1 <= sample_subject <= NUM_SUBJECTS:
            raise ValueError("Sample ID must be between 1 and 100.")
        if frequency not in TEST_FREQUENCIES:
            raise ValueError("Select one of the available held-out frequencies.")

        trial = self.trials[(sample_subject, frequency)]
        embeddings, _ = self._extract([trial])
        score = float(np.dot(embeddings[0], self.templates[profile_subject]))
        challenge = secrets.token_hex(24)
        return {
            "accepted": score >= self.threshold,
            "score": score,
            "threshold": self.threshold,
            "profile": profile_subject,
            "sample_subject": sample_subject,
            "frequency": frequency,
            "session": TEST_SESSION,
            "challenge": challenge,
        }


class WalletDemoHandler(SimpleHTTPRequestHandler):
    auth_service = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self):
        if urlparse(self.path).path in (
            "/",
            "/index.html",
            "/app.js",
            "/style.css",
        ):
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/vendor/ethers.umd.min.js":
            self._serve_ethers()
            return
        if request_path == "/api/status":
            self._send_json(
                200,
                {
                    "ready": True,
                    "threshold": self.auth_service.threshold,
                    "calibration_eer": self.auth_service.calibration_eer,
                    "channels": self.auth_service.checkpoint.get(
                        "channel_names", []
                    ),
                    "training_sessions": self.auth_service.training_sessions,
                    "training_frequencies": self.auth_service.training_frequencies,
                    "training_trials": sum(
                        int(item["session"]) in self.auth_service.training_sessions
                        and float(item["frequency"])
                        in self.auth_service.training_frequencies
                        for item in self.auth_service.metadata
                    ),
                    "subject_count": NUM_SUBJECTS,
                    "enrollment_session": ENROLLMENT_SESSION,
                    "enrollment_frequencies": self.auth_service.training_frequencies,
                    "calibration_session": CALIBRATION_SESSION,
                    "calibration_frequencies": TEST_FREQUENCIES,
                    "test_frequencies": TEST_FREQUENCIES,
                    "test_session": TEST_SESSION,
                    "chain": "Base Sepolia",
                    "demo_only": True,
                },
            )
            return
        super().do_GET()

    def _serve_ethers(self):
        bundle = (
            Path(__file__).resolve().parent.parent
            / "node_modules"
            / "ethers"
            / "dist"
            / "ethers.umd.min.js"
        )
        if not bundle.is_file():
            self._send_json(
                503,
                {
                    "error": (
                        "Local ethers browser bundle is missing. Run "
                        "'npm install' from the project root."
                    )
                },
            )
            return
        content = bundle.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/javascript; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if urlparse(self.path).path != "/api/authenticate":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request length."})
            return
        if content_length < 1 or content_length > 4096:
            self._send_json(413, {"error": "Request body size is invalid."})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object.")
            profile_subject = int(payload["profile_subject"])
            sample_subject = int(payload["sample_subject"])
            frequency = float(payload["frequency"])
            result = self.auth_service.authenticate(
                profile_subject,
                sample_subject,
                frequency,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
            return
        self._send_json(200, result)

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        print("wallet-demo:", format_string % args)


def main():
    parser = argparse.ArgumentParser(
        description="Run a dataset-backed EEG wallet demo on localhost."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=CACHE_ROOT / "eeg_embedding_8ch.pt",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("The demo server must bind to localhost only.")
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"Wallet demo web files not found: {WEB_ROOT}")

    WalletDemoHandler.auth_service = DemoAuthService(args.model)
    server = ThreadingHTTPServer((args.host, args.port), WalletDemoHandler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local wallet demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
