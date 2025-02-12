# Next.js Project Setup and Backend Integration

## Step-by-Step Instructions

### Step 1: Clone the Repository

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/mehul79/Vayu-Rakshak.git
   cd Vayu-Rakshak
   ```

---

### Step 2: Set Up the Backend

1. **Create a Virtual Environment** (Recommended for Python projects):
   ```sh
   python -m venv hackfiesta
   ```

2. **Activate the Virtual Environment**:
   - On Windows:
     ```sh
     hackfiesta\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source hackfiesta/bin/activate
     ```

3. **Start the Backend Server**:
   ```sh
   python3 main.py
   ```
   or
   ```sh
   python main.py
   ```

5. **Test the Backend API**:
   Open your browser or use Postman to send a POST request with a video file to:
   ```
   http://127.0.0.1:8000/predict/
   ```

---

### Step 3: Set Up the Frontend (Next.js)

#### Prerequisites
Ensure you have the following installed:
- [Node.js](https://nodejs.org/) (Recommended: Latest LTS version)
- [npm](https://www.npmjs.com/)
#### Installation

1. **Install Dependencies**:
   ```sh
   npm install
   ```

#### Running the Development Server

Start the development server with:
   ```sh
   npm run dev
   ```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

#### Environment Variables

Create a `.env` file in the root directory and add necessary environment variables:
   ```sh
   # API Endpoints
   NEXT_PUBLIC_ML_SERVICE_URL=http://localhost:8000
   NEXT_PUBLIC_VIDEO_STORAGE_PATH=../hackfiesta_backend_ml/processed_videos

   # App Configuration
   NEXT_PUBLIC_APP_NAME=Vayu Rakshak

   # Demo Videos
   NEXT_PUBLIC_DEMO_BEFORE_VIDEO=/videos/before.mp4
   NEXT_PUBLIC_DEMO_AFTER_VIDEO=/videos/after.mov 
   ```

