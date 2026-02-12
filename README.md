<h1 align="center">🔐 YHPARGONAGETS: Steganography & Cryptography Toolkit 🎨</h1>

<div align="center">
  <table>
    <tr>
      <td width="55%">
        <h3><b>About the Project</b></h3>
        <p>
          <b>YHPARGONAGETS</b> is a powerful Python-based toolkit that blends <strong>cryptographic encryption</strong> with <strong>steganographic techniques</strong>
          to secure messages in plain sight. Whether hiding secrets in whitespace or applying encryption before obfuscation, this project
          demonstrates the fascinating world of <code>secure communication</code> using visual and invisible mediums.
        </p>
      </td>
      <td width="45%">
        <img src="https://user-images.githubusercontent.com/74038190/235224431-e8c8c12e-6826-47f1-89fb-2ddad83b3abf.gif" width="100%">
      </td>
    </tr>
  </table>
</div>

## 📁 Project Structure

```bash
YHPARGONAGETS
├── C_Graphy                     🔐 Cryptography Utilities
│   ├── 3DeS.py                  Triple DES algorithm
│   ├── AdEnSt.py                Advanced Encryption + Stego logic
│   ├── BlOwFiSh.py              Blowfish encryption module
│   ├── EcE.py                   ECC or similar encryption
│   └── RiShAd.py                Custom crypto implementation
│
├── S_Graphy                     🧠 Steganography Modules
│   ├── Formatting               📝 Formatting-based Steganography
│   │   ├── Formatting.py            Core script
│   │   └── format_log.json          Encoding log
│   │
│   ├── Synonym                  🧠 Synonym-based Steganography
│   │   └── Synonym.py               Synonym encoding script
│   │
│   └── Whitespace              🧙‍♂️ Whitespace Steganography
│       ├── whitespace_stegano.py       Core logic
│       ├── python whitespace_stegano_gui.py  GUI version (consider renaming)
│       ├── stego_log.json              Stego log file
│       ├── stego_log2.json             Alternate log
│       └── syn_stego_log.json          Possibly related to Synonym
│
└── README.md                   📘 Project documentation

```
# 📂 YHPARGONAGETS

**YHPARGONAGETS** is a combined project on **Cryptography** and **Steganography**, offering various techniques to hide or secure information using different algorithms and encoding methods. It demonstrates classic and modern approaches to both fields in a modular, script-based layout.

---

## 🔐 C_Graphy — Cryptography Utilities

This folder contains implementations of popular and custom encryption algorithms:

- `3DeS.py` – Implements the Triple DES encryption algorithm  
- `AdEnSt.py` – A combined logic of Advanced Encryption + Steganography  
- `BlOwFiSh.py` – Blowfish cipher implementation  
- `EcE.py` – Likely ECC (Elliptic Curve Cryptography) or similar encryption method  
- `RiShAd.py` – A custom or experimental encryption method  

---

## 🧠 S_Graphy — Steganography Modules

This directory contains different steganography techniques used to conceal information within text using whitespace, formatting, or synonyms.

### 📝 Formatting/
- `Formatting.py` – Encodes binary data using text formatting (e.g., extra spaces)  
- `format_log.json` – Stores the log of formatting operations and results  

### 🧠 Synonym/
- `Synonym.py` – Replaces words with synonyms to hide binary data within readable text  

### 🧙‍♂️ Whitespace/
- `whitespace_stegano.py` – Core script using whitespace (single vs double spaces) for encoding  
- `python whitespace_stegano_gui.py` – GUI version of the whitespace steganography script  
- `stego_log.json` – Log file storing basic whitespace encoding operations  
- `stego_log2.json` – Variant or extended log file  
- `syn_stego_log.json` – May be related to synonym steganography (consider relocating if needed)  

---

## 📘 README.md
This file provides an overview of the entire project, directory structure, and description of scripts and utilities.

---

## ✅ Features

- Multiple cryptographic algorithm implementations  
- Text-based steganography using:
  - Whitespace variations  
  - Synonym substitution  
  - Formatting differences  
- Logging and retrieval of hidden messages  
- Support for multiple encoding/decoding sessions  
- Easy-to-extend modular design  

---

## 🚀 Getting Started

Follow the steps to run your own steganography & cryptography magic:

```bash
# Clone the repository
git clone https://github.com/your-username/YHPARGONAGETS.git
cd YHPARGONAGETS

# For whitespace stego tool
cd S_Graphy/Whitespace
python whitespace_stegano.py

# For cryptography
cd ../../C_Graphy
python AdEnSt.py
```

> 💡 Make sure Python 3.x is installed.

---
## 🎥 Project Demo

![Steganography Demo](https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif)


## 🧰 Tech Stack Used

| Category         | Tools/Libs       |
|------------------|------------------|
| 🐍 Language       | Python 3         |
| 📁 IDE           | VS Code          |
| 🔐 Concepts       | Steganography, Cryptography |
| ⚙️ Modules       | `os`, `pycryptodome`, etc. |

---

## 👨‍💻 Contributors

| Name              | GitHub Profile                                  |
|-------------------|-------------------------------------------------|
| Debayan Ghosh    | [@Debayan-Ghosh2005](https://github.com/Debayan-Ghosh2005) |
| Nirnoy Chatterjee  | [@Nirnoy12](https://github.com/Nirnoy12)|
| Sulagna Chakrabarty  | [@celestial201](https://github.com/celestial201)|
| Subhradeep Kar  | [@SubOptimal](https://github.com/SubOptimal-Official)|
| Sagnik Chakraborty  | [@evil-dodo8144](https://github.com/evil-dodo8144)|
---
## 🙌 Contribute - Only for invited accounts!

You might encounter some bugs while using this app. You are more than welcome to contribute. Just submit changes via pull request and I will review them before merging. Make sure you follow community guidelines.

## ⭐ Give A Star

You can also give this repository a star to show more people and they can use this repository.ok!
## 📊 GitHub Insights

![GitHub last commit](https://img.shields.io/github/last-commit/Debayan-Ghosh2005/yhpargonagets)
![GitHub issues](https://img.shields.io/github/issues/Debayan-Ghosh2005/yhpargonagets)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Debayan-Ghosh2005/yhpargonagets)

---


## 📜 License

This project is licensed under the **MIT License**.  
Feel free to fork, use, and improve!
Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

<a href="https://github.com/Debayan-Ghosh2005/yhpargonagets">
  <img src="https://forthebadge.com/images/badges/built-with-love.svg" width="130" alt="made with love markdown badge">
</a>
<a href="https://github.com/Debayan-Ghosh2005/yhpargonagets">
  <img src="https://forthebadge.com/images/badges/built-with-swag.svg" width="130" alt="made with swag markdown badge">
</a>
<a href="https://github.com/Debayan-Ghosh2005/yhpargonagets">
  <img src="https://forthebadge.com/images/badges/open-source.svg" width="130" height="30" alt="open source markdown badge">
</a>
<br>
<a href="https://github.com/Debayan-Ghosh2005/yhpargonagets">
  <img src="https://forthebadge.com/images/badges/made-with-markdown.svg" width="230" height="30" alt="made with markdown badge">
</a>

---

<h3 align="center">💫 Made with ❤️ by Team Friend 💫</h3>
