const BASE_SEPOLIA_CHAIN_ID = 84532n;
const BASE_SEPOLIA_RPC = "https://sepolia.base.org";
const BASE_SEPOLIA_EXPLORER = "https://sepolia.basescan.org";
const VAULT_STORAGE_KEY = "ssvep-demo-encrypted-wallet";
const PROFILE_STORAGE_PREFIX = "ssvep-demo-profile:";
const DATABASE_NAME = "ssvep-demo-wallet-vault";
const DATABASE_STORE = "keys";
const AES_KEY_ID = "wallet-wrapping-key";

const createWalletButton = document.querySelector("#create-wallet-button");
const lockWalletButton = document.querySelector("#lock-wallet-button");
const registerButton = document.querySelector("#register-button");
const authenticateButton = document.querySelector("#authenticate-button");
const signButton = document.querySelector("#sign-button");
const transactionButton = document.querySelector("#transaction-button");
const walletState = document.querySelector("#wallet-state");
const keyDisplay = document.querySelector("#key-display");
const encryptedKeyValue = document.querySelector("#encrypted-key-value");
const decryptedKeyPanel = document.querySelector("#decrypted-key-panel");
const decryptedKeyValue = document.querySelector("#decrypted-key-value");
const profileState = document.querySelector("#profile-state");
const authResult = document.querySelector("#auth-result");
const actionState = document.querySelector("#action-state");
const profileSelect = document.querySelector("#profile-subject");
const sampleSubjectSelect = document.querySelector("#sample-subject");
const frequencySelect = document.querySelector("#frequency");
const modelStatus = document.querySelector("#model-status");
const trainingSplit = document.querySelector("#training-split");
const trainingCount = document.querySelector("#training-count");
const enrollmentSplit = document.querySelector("#enrollment-split");
const calibrationSplit = document.querySelector("#calibration-split");
const testSplit = document.querySelector("#test-split");
const explorerLink = document.querySelector("#explorer-link");

let vault = null;
let enrolledProfile = null;
let unlockedWallet = null;
let authorization = null;
let signedChallenge = null;

for (let subject = 1; subject <= 100; subject += 1) {
  const option = new Option(
    `Dataset profile ${String(subject).padStart(3, "0")}`,
    subject,
  );
  profileSelect.add(option);
  sampleSubjectSelect.add(option.cloneNode(true));
}

function setStatus(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function bytesToBase64(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function base64ToBytes(value) {
  const binary = atob(value);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function openKeyDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(DATABASE_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function storeWrappingKey(key) {
  const database = await openKeyDatabase();
  try {
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(DATABASE_STORE, "readwrite");
      transaction.objectStore(DATABASE_STORE).put(key, AES_KEY_ID);
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally {
    database.close();
  }
}

async function loadWrappingKey() {
  const database = await openKeyDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(DATABASE_STORE, "readonly");
      const request = transaction.objectStore(DATABASE_STORE).get(AES_KEY_ID);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  } finally {
    database.close();
  }
}

async function clearWrappingKey() {
  const database = await openKeyDatabase();
  try {
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(DATABASE_STORE, "readwrite");
      transaction.objectStore(DATABASE_STORE).delete(AES_KEY_ID);
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally {
    database.close();
  }
}

function clearUnlockedState() {
  unlockedWallet = null;
  authorization = null;
  signedChallenge = null;
  decryptedKeyValue.textContent = "";
  decryptedKeyPanel.classList.add("hidden");
  signButton.disabled = true;
  transactionButton.disabled = true;
  lockWalletButton.disabled = true;
  explorerLink.classList.add("hidden");
}

function profileStorageKey(address) {
  return `${PROFILE_STORAGE_PREFIX}${address.toLowerCase()}`;
}

function refreshVaultUi() {
  const hasVault = Boolean(vault);
  registerButton.disabled = !hasVault;
  authenticateButton.disabled = !hasVault || !enrolledProfile;
  createWalletButton.textContent = hasVault
    ? "Replace local demo wallet"
    : "Create local wallet";

  if (!vault) {
    keyDisplay.classList.add("hidden");
    encryptedKeyValue.textContent = "";
    setStatus(walletState, "No local wallet created");
    setStatus(profileState, "Create a wallet, then enroll a dataset demo profile.");
    return;
  }

  const state = unlockedWallet ? "UNLOCKED IN PAGE MEMORY" : "ENCRYPTED IN THIS BROWSER";
  setStatus(
    walletState,
    `${state} · ${vault.address}`,
  );
  keyDisplay.classList.remove("hidden");
  encryptedKeyValue.textContent =
    `IV (Base64): ${vault.iv}\nCiphertext (Base64): ${vault.ciphertext}`;
  if (unlockedWallet) {
    decryptedKeyValue.textContent = unlockedWallet.privateKey;
    decryptedKeyPanel.classList.remove("hidden");
  }
  const savedProfile = localStorage.getItem(profileStorageKey(vault.address));
  if (savedProfile) {
    enrolledProfile = Number(savedProfile);
    profileSelect.value = String(enrolledProfile);
    setStatus(
      profileState,
      `Browser-local demo enrollment: wallet ↔ dataset profile ${String(enrolledProfile).padStart(3, "0")}.`,
    );
  } else {
    enrolledProfile = null;
    setStatus(profileState, "Choose a dataset profile and save the local demo mapping.");
  }
  authenticateButton.disabled = !enrolledProfile;
  lockWalletButton.disabled = !unlockedWallet;
}

async function createEncryptedWallet() {
  if (vault && !window.confirm(
    "Replace this browser's demo wallet? The old encrypted key will be deleted. Continue?",
  )) {
    return;
  }
  clearUnlockedState();
  if (vault) {
    localStorage.removeItem(VAULT_STORAGE_KEY);
    localStorage.removeItem(profileStorageKey(vault.address));
    await clearWrappingKey();
    vault = null;
  }

  const generatedWallet = ethers.Wallet.createRandom();
  const wrappingKey = await crypto.subtle.generateKey(
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const associatedData = new TextEncoder().encode(
    generatedWallet.address.toLowerCase(),
  );
  const plaintext = new TextEncoder().encode(generatedWallet.privateKey);
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv, additionalData: associatedData },
    wrappingKey,
    plaintext,
  );
  const savedVault = {
    address: generatedWallet.address,
    iv: bytesToBase64(iv),
    ciphertext: bytesToBase64(new Uint8Array(ciphertext)),
  };

  await storeWrappingKey(wrappingKey);
  localStorage.setItem(VAULT_STORAGE_KEY, JSON.stringify(savedVault));
  vault = savedVault;
  refreshVaultUi();
  setStatus(
    actionState,
    "Created a random test wallet. Its encrypted key is stored in this browser; it has no testnet funds yet.",
  );
}

async function unlockEncryptedWallet() {
  if (!vault) throw new Error("Create a local wallet first.");
  const wrappingKey = await loadWrappingKey();
  if (!wrappingKey) {
    throw new Error(
      "The local wrapping key is missing. The encrypted wallet cannot be unlocked; create a new demo wallet.",
    );
  }
  const plaintext = await crypto.subtle.decrypt(
    {
      name: "AES-GCM",
      iv: base64ToBytes(vault.iv),
      additionalData: new TextEncoder().encode(vault.address.toLowerCase()),
    },
    wrappingKey,
    base64ToBytes(vault.ciphertext),
  );
  const privateKey = new TextDecoder().decode(plaintext);
  const wallet = new ethers.Wallet(privateKey);
  if (wallet.address.toLowerCase() !== vault.address.toLowerCase()) {
    wallet.destroy();
    throw new Error("Decrypted key does not match the saved wallet address.");
  }
  unlockedWallet = wallet;
  decryptedKeyValue.textContent = wallet.privateKey;
  decryptedKeyPanel.classList.remove("hidden");
  refreshVaultUi();
}

createWalletButton.addEventListener("click", async () => {
  createWalletButton.disabled = true;
  try {
    await createEncryptedWallet();
  } catch (error) {
    setStatus(walletState, error.message || "Could not create the local wallet.", true);
  } finally {
    createWalletButton.disabled = false;
  }
});

lockWalletButton.addEventListener("click", () => {
  clearUnlockedState();
  refreshVaultUi();
  setStatus(actionState, "Wallet locked in this page. Refreshing the page also clears the in-memory wallet reference.");
});

registerButton.addEventListener("click", () => {
  if (!vault) return;
  enrolledProfile = Number(profileSelect.value);
  localStorage.setItem(
    profileStorageKey(vault.address),
    String(enrolledProfile),
  );
  authorization = null;
  signedChallenge = null;
  signButton.disabled = true;
  transactionButton.disabled = true;
  refreshVaultUi();
  setStatus(
    profileState,
    `Saved demo-only mapping: local wallet ↔ dataset profile ${String(enrolledProfile).padStart(3, "0")}.`,
  );
  setStatus(actionState, "Run the EEG model check to attempt to unlock this wallet.");
});

authenticateButton.addEventListener("click", async () => {
  if (!vault || !enrolledProfile) return;
  authenticateButton.disabled = true;
  clearUnlockedState();
  authResult.className = "result hidden";
  setStatus(actionState, "Running model inference on the selected public recording…");
  try {
    const response = await fetch("/api/authenticate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile_subject: enrolledProfile,
        sample_subject: Number(sampleSubjectSelect.value),
        frequency: Number(frequencySelect.value),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Model request failed.");

    if (!result.accepted) {
      authorization = null;
      authResult.className = "result fail";
      authResult.textContent =
        `MODEL REJECTED · score ${result.score.toFixed(6)} < threshold ${result.threshold.toFixed(6)}. The encrypted key remains locked.`;
      authResult.classList.remove("hidden");
      setStatus(actionState, "Authentication rejected; no wallet key was decrypted.");
      return;
    }

    await unlockEncryptedWallet();
    authorization = result;
    authResult.className = "result pass";
    authResult.textContent =
      `MODEL MATCH · score ${result.score.toFixed(6)} ≥ threshold ${result.threshold.toFixed(6)}. The encrypted wallet key was decrypted in this page's memory.`;
    authResult.classList.remove("hidden");
    signButton.disabled = false;
    lockWalletButton.disabled = false;
    setStatus(
      actionState,
      `Wallet unlocked in page memory: ${vault.address}. Choose “Sign with unlocked key” to approve the challenge.`,
    );
  } catch (error) {
    clearUnlockedState();
    setStatus(actionState, error.message || "Authentication or wallet unlock failed.", true);
  } finally {
    authenticateButton.disabled = false;
  }
});

signButton.addEventListener("click", async () => {
  if (!authorization || !unlockedWallet) return;
  const message = [
    "NeuroKey research demo wallet unlock",
    `Address: ${vault.address}`,
    "Chain ID: 84532",
    `Dataset profile: ${authorization.profile}`,
    `Frequency: ${authorization.frequency} Hz`,
    `Model score: ${authorization.score.toFixed(6)}`,
    `Challenge: ${authorization.challenge}`,
    "Public dataset replay; not a secure biometric assertion.",
  ].join("\n");
  try {
    const signature = await unlockedWallet.signMessage(message);
    const recoveredAddress = ethers.verifyMessage(message, signature);
    if (recoveredAddress.toLowerCase() !== vault.address.toLowerCase()) {
      throw new Error("Challenge signature did not recover the demo wallet.");
    }
    signedChallenge = signature;
    transactionButton.disabled = false;
    setStatus(
      actionState,
      `Challenge signature verified locally (${signature.slice(0, 14)}…). You may separately confirm the zero-value Base Sepolia transaction.`,
    );
  } catch (error) {
    setStatus(actionState, error.message || "Challenge signing failed.", true);
  }
});

transactionButton.addEventListener("click", async () => {
  if (!authorization || !signedChallenge || !unlockedWallet) return;
  if (!window.confirm(
    "Send a zero-value self-transfer from this local demo wallet on Base Sepolia? Network gas is required.",
  )) {
    return;
  }
  transactionButton.disabled = true;
  try {
    const provider = new ethers.JsonRpcProvider(BASE_SEPOLIA_RPC, {
      name: "base-sepolia",
      chainId: Number(BASE_SEPOLIA_CHAIN_ID),
    });
    const network = await provider.getNetwork();
    if (network.chainId !== BASE_SEPOLIA_CHAIN_ID) {
      throw new Error(`RPC returned unexpected chain ID ${network.chainId}.`);
    }
    const connectedWallet = unlockedWallet.connect(provider);
    const transaction = await connectedWallet.sendTransaction({
      to: vault.address,
      value: 0n,
      chainId: BASE_SEPOLIA_CHAIN_ID,
    });
    setStatus(
      actionState,
      `Broadcast with the EEG-unlocked local key. Waiting for confirmation: ${transaction.hash}`,
    );
    const receipt = await transaction.wait(1);
    if (!receipt || receipt.status !== 1) {
      throw new Error("The testnet transaction did not confirm successfully.");
    }
    explorerLink.href = `${BASE_SEPOLIA_EXPLORER}/tx/${transaction.hash}`;
    explorerLink.textContent = `View confirmed transaction ${transaction.hash.slice(0, 14)}… on BaseScan`;
    explorerLink.classList.remove("hidden");
    setStatus(
      actionState,
      `Confirmed on Base Sepolia. The local EEG check decrypted and used the demo key; the chain received only the zero-value self-transfer.`,
    );
  } catch (error) {
    setStatus(actionState, error.message || "Testnet transaction failed.", true);
  } finally {
    transactionButton.disabled = !signedChallenge;
  }
});

profileSelect.addEventListener("change", () => {
  registerButton.textContent = enrolledProfile ? "Change profile" : "Enroll profile";
});

sampleSubjectSelect.addEventListener("change", () => {
  clearUnlockedState();
  authResult.className = "result hidden";
  setStatus(actionState, "Sample changed; run EEG authentication again to unlock the key.");
});

frequencySelect.addEventListener("change", () => {
  clearUnlockedState();
  authResult.className = "result hidden";
  setStatus(actionState, "Frequency changed; run EEG authentication again to unlock the key.");
});

async function restoreVault() {
  const stored = localStorage.getItem(VAULT_STORAGE_KEY);
  if (!stored) {
    refreshVaultUi();
    return;
  }
  try {
    const savedVault = JSON.parse(stored);
    if (
      typeof savedVault.address !== "string"
      || typeof savedVault.iv !== "string"
      || typeof savedVault.ciphertext !== "string"
    ) {
      throw new Error("Stored local wallet record is invalid.");
    }
    vault = savedVault;
    if (!(await loadWrappingKey())) {
      throw new Error(
        "The encrypted wallet is present but its browser-held AES key is missing.",
      );
    }
    refreshVaultUi();
  } catch (error) {
    vault = null;
    setStatus(walletState, error.message || "Could not restore the encrypted wallet.", true);
  }
}

fetch("/api/status")
  .then((response) => response.json())
  .then((status) => {
    const formatFrequencies = (frequencies) =>
      frequencies.map((frequency) => `${Number(frequency)} Hz`).join(", ");
    const trainingSessions = status.training_sessions
      .map((session) => `ses-${session}`)
      .join(", ");
    const trainingFrequencies = formatFrequencies(status.training_frequencies);
    const enrollmentFrequencies = formatFrequencies(status.enrollment_frequencies);
    const calibrationFrequencies = formatFrequencies(status.calibration_frequencies);
    const testFrequencies = formatFrequencies(status.test_frequencies);
    trainingSplit.textContent =
      `${trainingSessions} · ${trainingFrequencies}`;
    trainingCount.textContent =
      `${status.training_trials.toLocaleString()} cached trials across ${status.subject_count} labeled dataset subjects. These labels are the training classes.`;
    enrollmentSplit.textContent =
      `Session ${status.enrollment_session} · ${enrollmentFrequencies}`;
    calibrationSplit.textContent =
      `Session ${status.calibration_session} · ${calibrationFrequencies}`;
    testSplit.textContent =
      `Session ${status.test_session} · ${testFrequencies}`;
    modelStatus.textContent =
      `8-channel model · ${status.channels.join(", ")} · ${status.test_frequencies.length} replayable test frequencies · local-only decision`;
  })
  .catch(() => {
    modelStatus.textContent = "Local model service unavailable.";
    trainingSplit.textContent = "Checkpoint details unavailable";
    trainingCount.textContent =
      "Start the local wallet service to load model and dataset split details.";
  });

restoreVault().catch((error) => {
  setStatus(walletState, error.message || "Could not restore the local wallet.", true);
});
