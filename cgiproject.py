import os
# --- NVIDIA GPU HINTS ---
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glm
import numpy as np
from PIL import Image
import math
import ctypes

# ==========================================
# 1. SHADERS (Modern Programmable Pipeline)
# ==========================================

# --- EARTH/MOON VERTEX SHADER ---
# (Can be shared because they both output Normal and TexCoords)
EARTH_V_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal;  
    TexCoords = aTexCoords;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# --- EARTH FRAGMENT SHADER ---
EARTH_F_SHADER = """
#version 330 core
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform sampler2D dayMap;
uniform sampler2D nightMap;
uniform sampler2D specMap;
uniform vec3 lightDir;
uniform vec3 viewPos;

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDirN = normalize(lightDir);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 ambient = 0.02 * vec3(texture(dayMap, TexCoords));
    float diff = max(dot(norm, lightDirN), 0.0);
    vec3 diffuse = diff * vec3(texture(dayMap, TexCoords));

    vec3 halfDir = normalize(lightDirN + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 32.0);
    float specMask = texture(specMap, TexCoords).r;       
    vec3 specular = vec3(0.6) * spec * specMask;          

    float dayNightDot = dot(norm, lightDirN);
    float nightMix = smoothstep(0.1, -0.1, dayNightDot);
    vec3 night = vec3(texture(nightMap, TexCoords)) * nightMix;
    
    vec3 result = ambient + diffuse + specular + night;
    FragColor = vec4(result, 1.0);
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

# --- MOON FRAGMENT SHADER ---
MOON_F_SHADER = """
#version 330 core
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 Normal;
in vec2 TexCoords;

uniform sampler2D moonMap;
uniform vec3 lightDir;

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDirN = normalize(lightDir);

    // Simple diffuse lighting for the Moon (phases)
    vec3 ambient = 0.02 * vec3(texture(moonMap, TexCoords));
    float diff = max(dot(norm, lightDirN), 0.0);
    vec3 diffuse = diff * vec3(texture(moonMap, TexCoords));

    FragColor = vec4(ambient + diffuse, 1.0);
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0); // Moon does not glow/bloom
}
"""

# --- BASIC SHADERS (For Glowing Artificial Satellites) ---
BASIC_V_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

BASIC_F_SHADER = """
#version 330 core
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

uniform vec3 color;
uniform float brightness;

void main() {
    vec3 finalColor = color * brightness;
    FragColor = vec4(finalColor, 1.0);
    BrightColor = vec4(finalColor, 1.0); 
}
"""

# --- SKYBOX SHADERS ---
SKYBOX_V_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main() {
    TexCoords = aPos; 
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww; // Force depth to 1.0
}
"""

SKYBOX_F_SHADER = """
#version 330 core
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in vec3 TexCoords;
uniform samplerCube skybox;

void main() {    
    FragColor = texture(skybox, TexCoords);
    BrightColor = vec4(0.0, 0.0, 0.0, 1.0); 
}
"""

# --- BLUR & COMPOSITE SHADERS ---
BLUR_V_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;
out vec2 TexCoords;
void main() {
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
"""

BLUR_F_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D image;
uniform bool horizontal;
uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {             
    vec2 tex_offset = 1.0 / textureSize(image, 0);
    vec3 result = texture(image, TexCoords).rgb * weight[0];
    
    if(horizontal) {
        for(int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
            result += texture(image, TexCoords - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
        }
    } else {
        for(int i = 1; i < 5; ++i) {
            result += texture(image, TexCoords + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
            result += texture(image, TexCoords - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
        }
    }
    FragColor = vec4(result, 1.0);
}
"""

FINAL_F_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D scene;
uniform sampler2D bloomBlur;

void main() {             
    vec3 hdrColor = texture(scene, TexCoords).rgb;      
    vec3 bloomColor = texture(bloomBlur, TexCoords).rgb;
    hdrColor += bloomColor;
    vec3 result = vec3(1.0) - exp(-hdrColor * 1.0); 
    FragColor = vec4(result, 1.0);
}
"""

# ==========================================
# 2. CAMERA AND INPUT
# ==========================================
class Camera:
    def __init__(self):
        # Pulled the camera back even further to see the Moon's large orbit
        self.pos = glm.vec3(0.0, 5.0, 18.0) 
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = -90.0
        self.pitch = -15.0
        self.speed = 4.0
        self.sensitivity = 0.1
        self.first_mouse = True
        self.last_x = 640
        self.last_y = 360

    def process_keyboard(self, window, delta_time):
        velocity = self.speed * delta_time
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.pos += self.front * velocity
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.pos -= self.front * velocity
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.pos -= glm.normalize(glm.cross(self.front, self.up)) * velocity
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.pos += glm.normalize(glm.cross(self.front, self.up)) * velocity

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False
        xoffset = (xpos - self.last_x) * self.sensitivity
        yoffset = (self.last_y - ypos) * self.sensitivity
        self.last_x, self.last_y = xpos, ypos
        self.yaw += xoffset
        self.pitch = max(-89.0, min(89.0, self.pitch + yoffset))
        
        front = glm.vec3()
        front.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front.y = math.sin(glm.radians(self.pitch))
        front.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)

# ==========================================
# 3. GEOMETRY & UTILITIES
# ==========================================
def create_sphere(radius, sectors, stacks):
    vertices, indices = [], []
    for i in range(stacks + 1):
        stack_angle = math.pi / 2 - i * math.pi / stacks
        xy = radius * math.cos(stack_angle)
        z = radius * math.sin(stack_angle)
        for j in range(sectors + 1):
            sector_angle = j * 2 * math.pi / sectors
            x = xy * math.cos(sector_angle)
            y = xy * math.sin(sector_angle)
            nx, ny, nz = x / radius, y / radius, z / radius
            s, t = 1.0 - (j / sectors), 1.0 - (i / stacks)
            vertices.extend([x, y, z, nx, ny, nz, s, t])
            
    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1
        for j in range(sectors):
            if i != 0: indices.extend([k1, k2, k1 + 1])
            if i != (stacks - 1): indices.extend([k1 + 1, k2, k2 + 1])
            k1 += 1; k2 += 1
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def create_orbit_path(a, b, segments):
    vertices = []
    for i in range(segments):
        theta = 2.0 * math.pi * float(i) / float(segments)
        vertices.extend([a * math.cos(theta), 0.0, b * math.sin(theta)])
    return np.array(vertices, dtype=np.float32)

def create_quad():
    vertices = np.array([
        -1.0,  1.0,  0.0, 1.0,  -1.0, -1.0,  0.0, 0.0,   1.0, -1.0,  1.0, 0.0,
        -1.0,  1.0,  0.0, 1.0,   1.0, -1.0,  1.0, 0.0,   1.0,  1.0,  1.0, 1.0
    ], dtype=np.float32)
    vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))
    return vao

def load_texture(path, unit):
    try:
        image = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(image, dtype=np.uint8).flatten()
    except Exception as e:
        print(f"Failed to load {path}. Creating dummy.")
        img_data = np.full((1, 1, 3), 255, dtype=np.uint8).flatten()
        image = Image.fromarray(np.full((1, 1, 3), 255, dtype=np.uint8))

    tex_id = glGenTextures(1)
    glActiveTexture(unit)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    format_ = GL_RGB if image.mode == "RGB" else GL_RGBA
    glTexImage2D(GL_TEXTURE_2D, 0, format_, image.width, image.height, 0, format_, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex_id

def load_cubemap(faces):
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)
    for i in range(len(faces)):
        try:
            image = Image.open(faces[i]) 
            img_data = np.array(image, dtype=np.uint8).flatten()
            format_ = GL_RGB if image.mode == "RGB" else GL_RGBA
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format_, 
                         image.width, image.height, 0, format_, GL_UNSIGNED_BYTE, img_data)
        except Exception as e:
            print(f"Failed to load cubemap face {faces[i]}. Error: {e}")

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    return tex_id

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    SCR_WIDTH, SCR_HEIGHT = 1280, 720
    window = glfw.create_window(SCR_WIDTH, SCR_HEIGHT, "Earth, Moon & Satellites", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glEnable(GL_DEPTH_TEST)

    camera = Camera()
    def mouse_callback(w, x, y): camera.process_mouse(x, y)
    glfw.set_cursor_pos_callback(window, mouse_callback)

    # --- Compile All Shaders ---
    earth_shader = compileProgram(compileShader(EARTH_V_SHADER, GL_VERTEX_SHADER), compileShader(EARTH_F_SHADER, GL_FRAGMENT_SHADER))
    moon_shader  = compileProgram(compileShader(EARTH_V_SHADER, GL_VERTEX_SHADER), compileShader(MOON_F_SHADER, GL_FRAGMENT_SHADER))
    basic_shader = compileProgram(compileShader(BASIC_V_SHADER, GL_VERTEX_SHADER), compileShader(BASIC_F_SHADER, GL_FRAGMENT_SHADER))
    skybox_shader = compileProgram(compileShader(SKYBOX_V_SHADER, GL_VERTEX_SHADER), compileShader(SKYBOX_F_SHADER, GL_FRAGMENT_SHADER))
    blur_shader  = compileProgram(compileShader(BLUR_V_SHADER, GL_VERTEX_SHADER), compileShader(BLUR_F_SHADER, GL_FRAGMENT_SHADER))
    final_shader = compileProgram(compileShader(BLUR_V_SHADER, GL_VERTEX_SHADER), compileShader(FINAL_F_SHADER, GL_FRAGMENT_SHADER))

    # --- Setup Geometry ---
    earth_v, earth_i = create_sphere(2.0, 64, 64)
    moon_v, moon_i   = create_sphere(0.5, 32, 32) # The Moon is larger than a satellite
    satellite_v, satellite_i = create_sphere(0.08, 16, 16)
    
    # Normalized circle for scaling paths
    orbit_v = create_orbit_path(1.0, 1.0, 120) 
    quad_vao = create_quad()
    
    skybox_vertices = np.array([
        -1.0,  1.0, -1.0,  -1.0, -1.0, -1.0,   1.0, -1.0, -1.0,   1.0, -1.0, -1.0,   1.0,  1.0, -1.0,  -1.0,  1.0, -1.0,
        -1.0, -1.0,  1.0,  -1.0, -1.0, -1.0,  -1.0,  1.0, -1.0,  -1.0,  1.0, -1.0,  -1.0,  1.0,  1.0,  -1.0, -1.0,  1.0,
         1.0, -1.0, -1.0,   1.0, -1.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0, -1.0,   1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,  -1.0,  1.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0,  1.0,   1.0, -1.0,  1.0,  -1.0, -1.0,  1.0,
        -1.0,  1.0, -1.0,   1.0,  1.0, -1.0,   1.0,  1.0,  1.0,   1.0,  1.0,  1.0,  -1.0,  1.0,  1.0,  -1.0,  1.0, -1.0,
        -1.0, -1.0, -1.0,  -1.0, -1.0,  1.0,   1.0, -1.0, -1.0,   1.0, -1.0, -1.0,  -1.0, -1.0,  1.0,   1.0, -1.0,  1.0
    ], dtype=np.float32)

    def setup_mesh(vertices, indices=None):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        if indices is not None:
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * vertices.itemsize, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * vertices.itemsize, ctypes.c_void_p(6 * vertices.itemsize))
            glEnableVertexAttribArray(2)
        else:
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
        return vao, len(indices) if indices is not None else len(vertices) // 3

    earth_vao, earth_count = setup_mesh(earth_v, earth_i)
    moon_vao, moon_count   = setup_mesh(moon_v, moon_i)
    sat_vao, sat_count     = setup_mesh(satellite_v, satellite_i)
    orbit_vao, orbit_count = setup_mesh(orbit_v)
    skybox_vao, _          = setup_mesh(skybox_vertices)

    # --- Load Textures ---
    tex_day = load_texture("day.jpg", GL_TEXTURE0)
    tex_night = load_texture("night.jpg", GL_TEXTURE1)
    tex_spec = load_texture("spec.jpg", GL_TEXTURE2)
    tex_moon = load_texture("moon.jpg", GL_TEXTURE3) # Add the Moon texture
    
    faces = ["right.jpg", "left.jpg", "top.jpg", "bottom.jpg", "front.jpg", "back.jpg"]
    cubemap_texture = load_cubemap(faces)

    # ==========================================
    # ARTIFICIAL SATELLITE CONFIGURATIONS
    # ==========================================
    satellites = [
        {"a": 2.5, "b": 2.5, "speed": 1.5, "tilt_x": 90.0, "tilt_y": 0.0, "color": (0.0, 0.8, 1.0), "phase": 0.0},
        {"a": 5.0, "b": 5.0, "speed": 0.3, "tilt_x": 0.0,  "tilt_y": 0.0, "color": (1.0, 0.8, 0.2), "phase": 1.5},
        {"a": 4.5, "b": 2.2, "speed": 0.7, "tilt_x": 45.0, "tilt_y": 30.0, "color": (1.0, 0.2, 0.5), "phase": 3.14},
        {"a": 3.0, "b": 6.0, "speed": 0.5, "tilt_x": -30.0, "tilt_y": 60.0, "color": (0.2, 1.0, 0.3), "phase": 2.0}
    ]

    # Moon Specific Params
    moon_distance = 9.0
    moon_speed = 0.1

    # ==========================================
    # FRAMEBUFFERS SETUP (HDR & Bloom)
    # ==========================================
    hdrFBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO)

    colorBuffers = glGenTextures(2)
    for i in range(2):
        glBindTexture(GL_TEXTURE_2D, colorBuffers[i])
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorBuffers[i], 0)

    attachments = (GLuint * 2)(GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1)
    glDrawBuffers(2, attachments)

    rboDepth = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    pingpongFBO = glGenFramebuffers(2)
    pingpongColorbuffers = glGenTextures(2)
    for i in range(2):
        glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[i])
        glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[i])
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingpongColorbuffers[i], 0)

    # ==========================================
    # RENDER LOOP
    # ==========================================
    sun_dir = glm.normalize(glm.vec3(1.0, 0.0, 0.5))
    
    glUseProgram(earth_shader)
    glUniform1i(glGetUniformLocation(earth_shader, "dayMap"), 0)
    glUniform1i(glGetUniformLocation(earth_shader, "nightMap"), 1)
    glUniform1i(glGetUniformLocation(earth_shader, "specMap"), 2)

    glUseProgram(moon_shader)
    glUniform1i(glGetUniformLocation(moon_shader, "moonMap"), 3)

    glUseProgram(skybox_shader)
    glUniform1i(glGetUniformLocation(skybox_shader, "skybox"), 0)

    glUseProgram(blur_shader)
    glUniform1i(glGetUniformLocation(blur_shader, "image"), 0)
    
    glUseProgram(final_shader)
    glUniform1i(glGetUniformLocation(final_shader, "scene"), 0)
    glUniform1i(glGetUniformLocation(final_shader, "bloomBlur"), 1)

    delta_time = 0.0
    last_frame = glfw.get_time()

    while not glfw.window_should_close(window):
        current_frame = glfw.get_time()
        delta_time, last_frame = current_frame - last_frame, current_frame

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        camera.process_keyboard(window, delta_time)

        glEnable(GL_DEPTH_TEST) 

        # --- PASS 1: RENDER SCENE & SKYBOX TO MRT FRAMEBUFFER ---
        glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO)
        glClearColor(0.01, 0.01, 0.02, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = glm.perspective(glm.radians(45.0), SCR_WIDTH / SCR_HEIGHT, 0.1, 100.0)
        view = glm.lookAt(camera.pos, camera.pos + camera.front, camera.up)

        # 1. Earth
        glUseProgram(earth_shader)
        glUniformMatrix4fv(glGetUniformLocation(earth_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(earth_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniform3fv(glGetUniformLocation(earth_shader, "lightDir"), 1, glm.value_ptr(sun_dir))
        glUniform3fv(glGetUniformLocation(earth_shader, "viewPos"), 1, glm.value_ptr(camera.pos))

        model = glm.rotate(glm.rotate(glm.mat4(1.0), glm.radians(23.5), glm.vec3(0, 0, 1)), current_frame * 0.2, glm.vec3(0, 1, 0))
        glUniformMatrix4fv(glGetUniformLocation(earth_shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
        normal_matrix = glm.mat3(glm.transpose(glm.inverse(model)))
        glUniformMatrix3fv(glGetUniformLocation(earth_shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normal_matrix))

        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_day)
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_night)
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_spec)
        glBindVertexArray(earth_vao)
        glDrawElements(GL_TRIANGLES, earth_count, GL_UNSIGNED_INT, None)

        # 2. The Moon
        glUseProgram(moon_shader)
        glUniformMatrix4fv(glGetUniformLocation(moon_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(moon_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniform3fv(glGetUniformLocation(moon_shader, "lightDir"), 1, glm.value_ptr(sun_dir))
        
        # Calculate Moon Position
        moon_x = moon_distance * math.cos(current_frame * moon_speed)
        moon_z = moon_distance * math.sin(current_frame * moon_speed)
        
        # Moon slightly tilted, rotating on its own axis
        moon_model = glm.translate(glm.mat4(1.0), glm.vec3(moon_x, 0.5, moon_z))
        moon_model = glm.rotate(moon_model, glm.radians(5.0), glm.vec3(0, 0, 1)) # Axial tilt
        moon_model = glm.rotate(moon_model, current_frame * 0.1, glm.vec3(0, 1, 0)) # Rotation
        
        glUniformMatrix4fv(glGetUniformLocation(moon_shader, "model"), 1, GL_FALSE, glm.value_ptr(moon_model))
        normal_matrix_moon = glm.mat3(glm.transpose(glm.inverse(moon_model)))
        glUniformMatrix3fv(glGetUniformLocation(moon_shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normal_matrix_moon))
        
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_moon)
        glBindVertexArray(moon_vao)
        glDrawElements(GL_TRIANGLES, moon_count, GL_UNSIGNED_INT, None)

        # 3. Render Artificial Satellites and Orbits
        glUseProgram(basic_shader)
        glUniformMatrix4fv(glGetUniformLocation(basic_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(basic_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        
        # Draw Moon's invisible orbit path just for visual reference
        moon_orbit_model = glm.scale(glm.translate(glm.mat4(1.0), glm.vec3(0, 0.5, 0)), glm.vec3(moon_distance, 1.0, moon_distance))
        glUniformMatrix4fv(glGetUniformLocation(basic_shader, "model"), 1, GL_FALSE, glm.value_ptr(moon_orbit_model))
        glUniform3f(glGetUniformLocation(basic_shader, "color"), 0.3, 0.3, 0.3) # Very subtle gray
        glUniform1f(glGetUniformLocation(basic_shader, "brightness"), 0.5)
        glBindVertexArray(orbit_vao)
        glDrawArrays(GL_LINE_LOOP, 0, orbit_count)

        # Loop through all artificial satellites
        for sat in satellites:
            orbit_model = glm.mat4(1.0)
            orbit_model = glm.rotate(orbit_model, glm.radians(sat["tilt_y"]), glm.vec3(0, 1, 0))
            orbit_model = glm.rotate(orbit_model, glm.radians(sat["tilt_x"]), glm.vec3(1, 0, 0))
            
            orbit_render_model = glm.scale(orbit_model, glm.vec3(sat["a"], 1.0, sat["b"]))
            glUniformMatrix4fv(glGetUniformLocation(basic_shader, "model"), 1, GL_FALSE, glm.value_ptr(orbit_render_model))
            glUniform3f(glGetUniformLocation(basic_shader, "color"), *sat["color"])
            glUniform1f(glGetUniformLocation(basic_shader, "brightness"), 1.5)

            glBindVertexArray(orbit_vao)
            glDrawArrays(GL_LINE_LOOP, 0, orbit_count)

            t = current_frame * sat["speed"] + sat["phase"]
            sat_local_pos = glm.vec3(sat["a"] * math.cos(t), 0.0, sat["b"] * math.sin(t))
            sat_world = glm.vec3(orbit_model * glm.vec4(sat_local_pos, 1.0))
            
            sat_model = glm.translate(glm.mat4(1.0), sat_world)
            glUniformMatrix4fv(glGetUniformLocation(basic_shader, "model"), 1, GL_FALSE, glm.value_ptr(sat_model))
            glUniform3f(glGetUniformLocation(basic_shader, "color"), *sat["color"])
            glUniform1f(glGetUniformLocation(basic_shader, "brightness"), 4.0)

            glBindVertexArray(sat_vao)
            glDrawElements(GL_TRIANGLES, sat_count, GL_UNSIGNED_INT, None)

        # 4. Skybox
        glDepthFunc(GL_LEQUAL)
        glUseProgram(skybox_shader)
        view_no_translation = glm.mat4(glm.mat3(view))
        glUniformMatrix4fv(glGetUniformLocation(skybox_shader, "view"), 1, GL_FALSE, glm.value_ptr(view_no_translation))
        glUniformMatrix4fv(glGetUniformLocation(skybox_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        
        glBindVertexArray(skybox_vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glDepthFunc(GL_LESS)

        glDisable(GL_DEPTH_TEST)

        # --- PASS 2: TWO-PASS GAUSSIAN BLUR ON BRIGHT BUFFER ---
        horizontal = True
        first_iteration = True
        amount = 10 
        glUseProgram(blur_shader)
        for i in range(amount):
            glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[int(horizontal)])
            glUniform1i(glGetUniformLocation(blur_shader, "horizontal"), horizontal)
            glActiveTexture(GL_TEXTURE0)
            
            glBindTexture(GL_TEXTURE_2D, colorBuffers[1] if first_iteration else pingpongColorbuffers[int(not horizontal)])
            
            glBindVertexArray(quad_vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            horizontal = not horizontal
            if first_iteration: first_iteration = False

        # --- PASS 3: COMPOSITE HDR SCENE + BLURRED BLOOM ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(final_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, colorBuffers[0]) 
        glActiveTexture(GL_TEXTURE1)
        
        glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[int(not horizontal)]) 
        
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()