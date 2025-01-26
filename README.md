<div style = "background-color: #ffefef; padding: 20px; color: #7D5C5C">

<h1>🌸 Eve: AI-Powered Feminine Product Scanner & Search 🌸</h1>

Eve is a web application designed to empower women with factual and easy-to-understand information about feminine products. Using **AI-powered analysis** and an intuitive **image scanner**, users can scan product packaging or search by name to learn about ingredients, side effects, and safer alternatives. 🚀✨

---

## 🌟 Features

- **📸 Product Scanner:**  
  Use your webcam to capture an image of a feminine product. The application identifies the product and provides detailed safety insights.

- **🔍 Intelligent Search:**  
  For a quick search, just enter the product name.

- **📤 Responses:**  
  Both the scanner and the search return the following information about the product:
  - Beneficial ingredients 🌿
  - Harmful ingredients ⚠️
  - Potential side effects 🩹
  - Safer alternatives 🌈
  - Safety ranking (1-5 scale) 🛡️

- **💬 AI-Powered Descriptions:**  
  Leverages **Groq API** to generate concise, easy-to-understand safety reports.

- **🖼️ Minimal UI Design:**  
  A clean, pink-themed interface for effortless navigation and an enjoyable experience. 🎀

---

## 🛠️ Installation & Setup

### 1️⃣ Prerequisites
- Python 3.8+
- Pip (Python package installer)
- Virtual environment (recommended)

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/tangalice/uterUS.git
cd eve-scanner
```

### 3️⃣ Install Dependencies
```bash
pip install groq python-dotenv flask numpy opencv-python keras base64 tensorflow==2.15.0
```

### 4️⃣ Add environment variables
Create a .env file and add yoru Groq API Key
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 5️⃣ Run the Application
This application will run on http://127.0.0.1:5004
```bash
python app.py
```

## 📚 Technologies Used

- **Flask**: A lightweight and powerful Python web framework for building the backend. 🌐  
- **Groq API**: An AI-powered API for generating insightful product analyses. 🤖  
- **Keras**: A deep learning library used for the pre-trained image classification model. 🧠  
- **OpenCV**: A computer vision library for image preprocessing and handling. 📸  
- **NumPy**: A library for numerical operations and data manipulation. 🔢  
- **HTML/CSS/JavaScript**: Frontend technologies used to create a user-friendly and pink-themed interface. 🎀  
- **dotenv**: For securely managing environment variables, including the Groq API key. 🔐  
</div>