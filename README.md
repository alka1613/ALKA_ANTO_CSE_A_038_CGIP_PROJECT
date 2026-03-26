
# OrbitRender: Earth, Moon & Satellites 🌍✨

A high-performance 3D celestial simulation built with **Python** and **Modern OpenGL (3.3+)**. This project features real-time HDR rendering, Gaussian bloom, and dynamic day/night transitions across Windows, macOS, and Linux.

---

## 🚀 Key Features

* **Modern OpenGL Pipeline:** Uses GLSL 3.30 shaders for all rendering, utilizing Vertex and Fragment shaders for a programmable pipeline.
* **HDR & Bloom:** Implements Multi-Render Targets (MRT) to separate high-brightness pixels and apply a two-pass Gaussian blur for a "glow" effect on artificial satellites.
* **Dynamic Earth Surface:** A custom fragment shader that mixes "Day" and "Night" textures based on the light source position, featuring a specular mask for realistic ocean reflections.
* **Orbital Mechanics:** Simulates complex orbits for the Moon and multiple satellites with varying speeds, inclinations (tilt), and phases using `glm` transformations.
* **Skybox Integration:** Implements a seamless cubemap background to provide an immersive deep-space environment.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Graphics API:** OpenGL 3.3 (Core Profile)
* **Libraries:**
    * `PyOpenGL`: Python bindings for the OpenGL API.
    * `GLFW`: Window management and user input handling.
    * `PyGLM`: Fast C++-style mathematics for matrix and vector operations.
    * `Pillow`: Image processing used for texture loading and mapping.

---

## 📦 Installation & Setup

### 1. System Prerequisites
Depending on your OS, you may need to install the GLFW library globally:

* **Windows:** No extra steps; the Python `glfw` package handles the necessary binaries.
* **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update
    sudo apt install libglfw3-dev libglew-dev
    ```
* **macOS:**
    ```bash
    brew install glfw
    ```

### 2. Install Dependencies
Ensure you have a `requirements.txt` file and run:
```bash
pip install -r requirements.txt
```

### 3. Manual Asset Setup
The program expects the following image files in the root directory alongside `cgiproject.py`:

| Category | File Names |
| :--- | :--- |
| **Earth** | `day.jpg`, `night.jpg`, `spec.jpg` |
| **Moon** | `moon.jpg` |
| **Skybox** | `right.jpg`, `left.jpg`, `top.jpg`, `bottom.jpg`, `front.jpg`, `back.jpg` |

### 4. Run the Simulation
```bash
python cgiproject.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **W / S** | Move Forward / Backward |
| **A / D** | Strafe Left / Right |
| **Mouse** | Look Around (Yaw/Pitch) |
| **ESC** | Close Application |

---

## 🏗️ Technical Deep Dive

### The Lighting Model
The Earth uses a modified Blinn-Phong model. The transition between the day-side and night-side textures (the terminator line) is handled by a `smoothstep` function in the fragment shader based on the dot product of the normal and light direction:

$$nightMix = smoothstep(0.1, -0.1, dot(Normal, lightDir))$$

### Bloom Pipeline (Post-Processing)
1.  **Render Pass:** The scene is rendered to an HDR Framebuffer with two color attachments: the full scene and a brightness-only buffer.
2.  **Blur Pass:** A 5-tap Gaussian blur is applied via "ping-ponging" between framebuffers to blur the bright areas.
3.  **Composite Pass:** The blurred glow is added back to the original scene, and tone mapping is applied to bring the colors back into a viewable range:
    $$result = 1.0 - exp(-hdrColor * 1.0)$$

---
