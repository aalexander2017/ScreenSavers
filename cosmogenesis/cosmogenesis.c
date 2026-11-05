/*
 * cosmogenesis - an OpenGL "history of the universe" screensaver.
 *
 * Scenes (looping):
 *   0. The Big Bang: fireball, cooling, dark ages, the cosmic web
 *   1. Birth of a spiral galaxy
 *   2. Two galaxies collide and merge
 *   3. The merged galaxy's supermassive black hole shreds a star
 *   4. A stellar nursery: gas clouds collapse and stars ignite
 *   5. A protoplanetary disk becomes a planetary system
 *   6. A red supergiant goes supernova, leaving a pulsar
 *
 * Rendering: OpenGL 3.3 shaders. Particles are simulated on the CPU and
 * streamed as one instance each; the GPU builds the sprites, applies the
 * black hole's gravitational lensing, lights the planets, and tone maps
 * an HDR buffer to the screen.
 *
 * Build:  gcc -O2 -o cosmogenesis cosmogenesis.c -lGL -lX11 -lm
 *
 * Usage:
 *   cosmogenesis               full screen, exits on key/mouse
 *   cosmogenesis -window       1280x720 window (space/n = next scene, q/Esc = quit)
 *   cosmogenesis -root         draw on root window / XSCREENSAVER_WINDOW (xscreensaver)
 *   cosmogenesis -scene N      start at scene N (0-6)
 *   cosmogenesis -speed X      time multiplier
 *   cosmogenesis -fps          print frame rate
 *   cosmogenesis -shot N T out.ppm   render scene N at time T to a PPM and exit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define GL_GLEXT_PROTOTYPES 1
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>

/* ------------------------------------------------------------------ */
/* Math helpers                                                        */
/* ------------------------------------------------------------------ */
#define PI 3.14159265f
static unsigned int rng = 0x12345678u;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng >> 8) * (1.0f / 16777216.0f); }
static float srand1(void) { return frand() * 2 - 1; }
static float gauss(void) { float u = frand() + 1e-7f, v = frand(); return sqrtf(-2 * logf(u)) * cosf(2 * PI * v); }
static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smooth(float e0, float e1, float x) { float t = clampf((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
static float lerp(float a, float b, float t) { return a + (b - a) * t; }
static float hash1(int n) { unsigned int h = (unsigned)n * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; return (h & 0xffffff) / 16777216.0f; }
static void rand_unit(float *d) { float z = srand1(), a = frand() * 2 * PI, r = sqrtf(1 - z * z); d[0] = r * cosf(a); d[1] = z; d[2] = r * sinf(a); }
static void norm3(float *v) { float l = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); if (l > 0) { v[0] /= l; v[1] /= l; v[2] /= l; } }
static void cross3(const float *a, const float *b, float *o) { o[0] = a[1]*b[2]-a[2]*b[1]; o[1] = a[2]*b[0]-a[0]*b[2]; o[2] = a[0]*b[1]-a[1]*b[0]; }
static void roty(float *x, float *z, float a) { float c = cosf(a), s = sinf(a), nx = *x * c - *z * s; *z = *x * s + *z * c; *x = nx; }

/* ------------------------------------------------------------------ */
/* Camera                                                              */
/* ------------------------------------------------------------------ */
static float V[16], Pm[16], camR[3], camU[3], camE[3], tanX, tanY;
static int scrW = 1280, scrH = 720;
static float gfade = 1;              /* global scene fade */

static void set_camera(const float *eye, const float *tgt, float fovdeg)
{
    float f[3] = { tgt[0]-eye[0], tgt[1]-eye[1], tgt[2]-eye[2] }, up[3] = { 0, 1, 0 }, s[3], u[3];
    norm3(f); cross3(f, up, s); norm3(s); cross3(s, f, u);
    memset(V, 0, sizeof V);
    V[0] = s[0]; V[4] = s[1]; V[8]  = s[2];
    V[1] = u[0]; V[5] = u[1]; V[9]  = u[2];
    V[2] = -f[0]; V[6] = -f[1]; V[10] = -f[2];
    V[12] = -(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2]);
    V[13] = -(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2]);
    V[14] =  (f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]);
    V[15] = 1;
    memcpy(camR, s, sizeof s); memcpy(camU, u, sizeof u); memcpy(camE, eye, sizeof camE);
    float aspect = (float)scrW / scrH, n = 0.05f, fa = 6000;
    tanY = tanf(fovdeg * PI / 360); tanX = tanY * aspect;
    memset(Pm, 0, sizeof Pm);
    Pm[0] = 1 / tanX; Pm[5] = 1 / tanY; Pm[10] = -(fa + n) / (fa - n); Pm[11] = -1; Pm[14] = -2 * fa * n / (fa - n);
}

/* ------------------------------------------------------------------ */
/* Batches. A sprite is one instance: pos.xyz, size, rgba, flags.      */
/* The GPU expands it into a camera-facing quad, fades it near the     */
/* camera and (in the black hole scene) bends it through the lens.     */
/* ------------------------------------------------------------------ */
typedef struct { float *d; int n, cap; GLuint vbo; } Batch;
static Batch glow, disc, front, lines;
#define SPR_FLOATS 9
#define LINE_FLOATS 14

static void batch_init(Batch *b, int cap, int floats)
{
    b->cap = cap; b->n = 0; b->d = malloc(sizeof(float) * floats * cap);
    glGenBuffers(1, &b->vbo);
}

static void emit_f(Batch *b, float x, float y, float z, float r, float g, float bl, float a, float size, float flag)
{
    if (b->n >= b->cap || a < 0.002f) return;
    float *d = b->d + SPR_FLOATS * b->n++;
    d[0] = x; d[1] = y; d[2] = z; d[3] = size; d[4] = r; d[5] = g; d[6] = bl; d[7] = a; d[8] = flag;
}
#define emit(b, x, y, z, r, g, bl, a, s)    emit_f(b, x, y, z, r, g, bl, a, s, 0.0f)
#define emit_nl(b, x, y, z, r, g, bl, a, s) emit_f(b, x, y, z, r, g, bl, a, s, 1.0f)   /* never lensed */

static void line(float x0, float y0, float z0, float x1, float y1, float z1, float r, float g, float b, float a)
{
    if (lines.n >= lines.cap) return;
    float *d = lines.d + LINE_FLOATS * lines.n++;
    d[0] = x0; d[1] = y0; d[2] = z0; d[3] = r; d[4] = g; d[5] = b; d[6] = a;
    d[7] = x1; d[8] = y1; d[9] = z1; d[10] = r; d[11] = g; d[12] = b; d[13] = a;
}

/* ------------------------------------------------------------------ */
/* Gravitational lensing (parameters; the maths is in the shader)      */
/* ------------------------------------------------------------------ */
static int lensOn = 0;
static float lensRs = 1, lensX, lensY, lensD, lensShadow = 2.6f;

static void lens_setup(const float *w, float rs)
{
    lensOn = 1; lensRs = rs; lensShadow = 2.6f * rs;
    lensX = V[0]*w[0] + V[4]*w[1] + V[8]*w[2] + V[12];
    lensY = V[1]*w[0] + V[5]*w[1] + V[9]*w[2] + V[13];
    lensD = -(V[2]*w[0] + V[6]*w[1] + V[10]*w[2] + V[14]);
}

/* ------------------------------------------------------------------ */
/* Shaders                                                             */
/* ------------------------------------------------------------------ */
static const char *spriteVS =
"#version 330 core\n"
"layout(location=0) in vec2 corner;\n"
"layout(location=1) in vec4 posSize;\n"
"layout(location=2) in vec4 color;\n"
"layout(location=3) in float flags;\n"
"uniform mat4 V, P;\n"
"uniform float fade;\n"
"uniform int lensMode;          /* 0 off, 1 primary image, 2 secondary image, 3 in front of lens */\n"
"uniform vec3 lensC;            /* lens in view space: x, y, depth */\n"
"uniform float lensRs, lensShadow;\n"
"uniform vec3 lightW;\n"
"uniform float pixScale, sizeMin, sizeMax;   /* draw only sprites in this on-screen size range */\n"
"out vec2 uv; out vec4 col; out vec3 ldir;\n"
"void main() {\n"
"  vec4 vp = V * vec4(posSize.xyz, 1.0);\n"
"  float depth = -vp.z, size = posSize.w, a = color.a * fade;\n"
"  bool nolens = flags > 0.5;\n"
"  if (lensMode != 0) {\n"
"    bool infront = depth <= lensC.z;\n"
"    if (nolens) { if (lensMode != 1) a = 0.0; }\n"
"    else if (lensMode == 3) { if (!infront) a = 0.0; }\n"
"    else if (infront) a = 0.0;\n"
"    else {\n"
"      float Dl = lensC.z, Ds = depth, RE2 = 2.4 * lensRs * Dl * (Ds - Dl) / Ds;\n"
"      vec2 b = vp.xy * Dl / Ds - lensC.xy; float beta = length(b) + 1e-5;\n"
"      float sq = sqrt(beta * beta + 4.0 * RE2), u = beta / sqrt(RE2 + 1e-9);\n"
"      float mu0 = (u * u + 2.0) / (2.0 * u * sqrt(u * u + 4.0));\n"
"      float th = lensMode == 1 ? 0.5 * (beta + sq) : 0.5 * (beta - sq);\n"
"      float mu = lensMode == 1 ? mu0 + 0.5 : mu0 - 0.5;\n"
"      if (abs(th) < lensShadow || mu < 0.02) a = 0.0;\n"
"      float sm = sqrt(min(mu, 2.5));\n"
"      vp.xy = (lensC.xy + b / beta * th) * Ds / Dl;\n"
"      a *= sm; size *= sm;\n"
"    }\n"
"  }\n"
"  a *= clamp((depth - 0.4 * size) / (1.5 * size + 0.5), 0.0, 1.0);\n"
"  float pix = size * pixScale / max(depth, 0.05);\n"
"  if (pix < sizeMin || pix >= sizeMax) a = 0.0;\n"
"  if (depth < 0.05 || a < 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); col = vec4(0.0); uv = vec2(0.0); ldir = vec3(0.0); return; }\n"
"  gl_Position = P * (vp + vec4(corner * size * 0.5, 0.0, 0.0));\n"
"  uv = corner; col = vec4(color.rgb, a);\n"
"  ldir = normalize(mat3(V) * (lightW - posSize.xyz));\n"
"}\n";

static const char *glowFS =
"#version 330 core\n"
"in vec2 uv; in vec4 col; in vec3 ldir; out vec4 o;\n"
"void main() {\n"
"  float r2 = dot(uv, uv); if (r2 > 1.0) discard;\n"
"  float r = sqrt(r2);\n"
"  float a = exp(-r2 * 5.5) * (1.0 - smoothstep(0.85, 1.0, r)) * 0.85 + 0.15 * exp(-r2 * 40.0);\n"
"  o = vec4(col.rgb * (col.a * a), 1.0);\n"
"}\n";

static const char *discFS =
"#version 330 core\n"
"in vec2 uv; in vec4 col; in vec3 ldir; out vec4 o;\n"
"void main() {\n"
"  float r2 = dot(uv, uv); if (r2 > 1.0) discard;\n"
"  vec3 n = vec3(uv, sqrt(max(0.0, 1.0 - r2)));\n"
"  float lam = max(dot(n, ldir), 0.0);\n"
"  float limb = mix(0.75, 1.0, n.z);\n"
"  float lit = 0.035 + 0.965 * pow(lam, 0.85) * limb;\n"
"  float edge = 1.0 - smoothstep(0.9, 1.0, sqrt(r2));\n"
"  o = vec4(col.rgb * lit, col.a * edge);\n"
"}\n";

static const char *lineVS =
"#version 330 core\n"
"layout(location=0) in vec3 pos; layout(location=1) in vec4 color;\n"
"uniform mat4 V, P; uniform float fade; out vec4 col;\n"
"void main() { gl_Position = P * V * vec4(pos, 1.0); col = vec4(color.rgb, color.a * fade); }\n";

static const char *lineFS =
"#version 330 core\n"
"in vec4 col; out vec4 o;\n"
"void main() { o = vec4(col.rgb * col.a, 1.0); }\n";

static const char *postVS =
"#version 330 core\n"
"out vec2 uv;\n"
"void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";

/* HDR -> display: linear below 0.8, soft roll-off above, plus flash and dither */
static const char *postFS =
"#version 330 core\n"
"in vec2 uv; out vec4 o;\n"
"uniform sampler2D hdr, hdrLow, hdrLow2; uniform vec4 flash;\n"
"void main() {\n"
"  vec3 c = texture(hdr, uv).rgb + texture(hdrLow, uv).rgb + texture(hdrLow2, uv).rgb + flash.rgb * flash.a;\n"
"  vec3 hi = 0.8 + 0.2 * (1.0 - exp(-(c - 0.8) / 0.2));\n"
"  c = mix(c, hi, step(0.8, c));\n"
"  c += (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;\n"
"  o = vec4(c, 1.0);\n"
"}\n";

typedef struct { GLuint id; GLint V, P, fade, mode, lensC, lensRs, lensSh, light, pix, smin, smax; } Prog;
static Prog progGlow, progDisc, progLine;
static GLuint progPost, postFlash, postTex, postLow, postLow2;
static GLuint vaoSprite, vaoLine, vaoPost, quadVbo, fbo, fboTex, fboLow, fboLowTex, fboLow2, fboLow2Tex;
static int fboW = 0, fboH = 0;
/* sprites are split by on-screen width: < 16 px full res, < 64 px half res, larger quarter res */
static const float layerMin[3] = { 0, 16, 64 }, layerMax[3] = { 16, 64, 1e9f };
static float sizeMin = 0, sizeMax = 1e9f;

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type); GLint ok;
    glShaderSource(s, 1, &src, NULL); glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s, sizeof log, NULL, log); fprintf(stderr, "cosmogenesis: shader error:\n%s\n", log); exit(1); }
    return s;
}

static GLuint link_prog(const char *vs, const char *fs)
{
    GLuint p = glCreateProgram(); GLint ok;
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs)); glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p); glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log); fprintf(stderr, "cosmogenesis: link error:\n%s\n", log); exit(1); }
    return p;
}

static Prog make_prog(const char *vs, const char *fs)
{
    Prog p; p.id = link_prog(vs, fs);
    p.V = glGetUniformLocation(p.id, "V"); p.P = glGetUniformLocation(p.id, "P");
    p.fade = glGetUniformLocation(p.id, "fade"); p.mode = glGetUniformLocation(p.id, "lensMode");
    p.lensC = glGetUniformLocation(p.id, "lensC"); p.lensRs = glGetUniformLocation(p.id, "lensRs");
    p.lensSh = glGetUniformLocation(p.id, "lensShadow"); p.light = glGetUniformLocation(p.id, "lightW");
    p.pix = glGetUniformLocation(p.id, "pixScale"); p.smin = glGetUniformLocation(p.id, "sizeMin"); p.smax = glGetUniformLocation(p.id, "sizeMax");
    return p;
}

static void gl_setup(void)
{
    progGlow = make_prog(spriteVS, glowFS);
    progDisc = make_prog(spriteVS, discFS);
    progLine = make_prog(lineVS, lineFS);
    progPost = link_prog(postVS, postFS);
    postFlash = glGetUniformLocation(progPost, "flash"); postTex = glGetUniformLocation(progPost, "hdr"); postLow = glGetUniformLocation(progPost, "hdrLow"); postLow2 = glGetUniformLocation(progPost, "hdrLow2");

    static const float quad[8] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    glGenVertexArrays(1, &vaoSprite); glBindVertexArray(vaoSprite);
    glGenBuffers(1, &quadVbo); glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    for (int i = 1; i <= 3; i++) { glEnableVertexAttribArray(i); glVertexAttribDivisor(i, 1); }
    glGenVertexArrays(1, &vaoLine); glBindVertexArray(vaoLine);
    glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);
    glGenVertexArrays(1, &vaoPost);
    glBindVertexArray(0);

    batch_init(&glow, 260000, SPR_FLOATS); batch_init(&disc, 256, SPR_FLOATS);
    batch_init(&front, 60000, SPR_FLOATS); batch_init(&lines, 4096, LINE_FLOATS);
    glDisable(GL_DEPTH_TEST);
}

static void make_target(GLuint *fb, GLuint *tex, int w, int h, GLint filter)
{
    if (*fb) { glDeleteFramebuffers(1, fb); glDeleteTextures(1, tex); }
    glGenTextures(1, tex); glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, fb); glBindFramebuffer(GL_FRAMEBUFFER, *fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "cosmogenesis: HDR framebuffer unsupported\n"); exit(1); }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void ensure_fbo(void)
{
    if (fboW == scrW && fboH == scrH) return;
    make_target(&fbo, &fboTex, scrW, scrH, GL_NEAREST);
    make_target(&fboLow, &fboLowTex, (scrW + 1) / 2, (scrH + 1) / 2, GL_LINEAR);
    make_target(&fboLow2, &fboLow2Tex, (scrW + 3) / 4, (scrH + 3) / 4, GL_LINEAR);
    fboW = scrW; fboH = scrH;
}

static float flashA = 0, flashR = 1, flashG = 1, flashB = 1;

static void upload(Batch *b, int floats)
{
    glBindBuffer(GL_ARRAY_BUFFER, b->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * floats * b->n, NULL, GL_STREAM_DRAW);   /* orphan */
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * floats * b->n, b->d);
}

static void use_prog(const Prog *p, int mode)
{
    glUseProgram(p->id);
    glUniformMatrix4fv(p->V, 1, GL_FALSE, V); glUniformMatrix4fv(p->P, 1, GL_FALSE, Pm);
    glUniform1f(p->fade, gfade);
    if (p->mode >= 0) {
        glUniform1i(p->mode, mode);
        glUniform3f(p->lensC, lensX, lensY, lensD);
        glUniform1f(p->lensRs, lensRs); glUniform1f(p->lensSh, lensShadow);
        glUniform3f(p->light, 0, 0, 0);    /* every lit body in the show orbits a star at the origin */
        glUniform1f(p->pix, Pm[0] * scrW * 0.5f); glUniform1f(p->smin, sizeMin); glUniform1f(p->smax, sizeMax);
    }
}

static void draw_sprites(Batch *b, const Prog *p, int mode)
{
    if (!b->n) return;
    use_prog(p, mode);
    glBindVertexArray(vaoSprite);
    glBindBuffer(GL_ARRAY_BUFFER, b->vbo);
    const GLsizei st = sizeof(float) * SPR_FLOATS;
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, st, (void *)0);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, st, (void *)(4 * sizeof(float)));
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, st, (void *)(8 * sizeof(float)));
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, b->n);
}

/* One layer of the frame. The full-resolution layer gets the small, sharp
 * sprites; the half-resolution layer gets the big soft ones, which is where
 * nearly all the overdraw is. Both see the same draw order. */
static void render_layer(int low)
{
    int dv = 1 << low;
    glBindFramebuffer(GL_FRAMEBUFFER, low == 2 ? fboLow2 : low ? fboLow : fbo);
    glViewport(0, 0, (scrW + dv - 1) / dv, (scrH + dv - 1) / dv);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    sizeMin = layerMin[low]; sizeMax = layerMax[low];

    glBlendFunc(GL_ONE, GL_ONE);
    if (lensOn) { draw_sprites(&glow, &progGlow, 1); draw_sprites(&glow, &progGlow, 2); }
    else draw_sprites(&glow, &progGlow, 0);

    if (lines.n && !low) {
        use_prog(&progLine, 0);
        glBindVertexArray(vaoLine); glBindBuffer(GL_ARRAY_BUFFER, lines.vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void *)(3 * sizeof(float)));
        glDrawArrays(GL_LINES, 0, lines.n * 2);
    }

    /* solid bodies: drawn at any size; in the low layer they only block light */
    sizeMin = 0; sizeMax = 1e9f;
    if (low) glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
    else glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    draw_sprites(&disc, &progDisc, 0);
    sizeMin = layerMin[low]; sizeMax = layerMax[low];

    glBlendFunc(GL_ONE, GL_ONE);
    if (lensOn) draw_sprites(&glow, &progGlow, 3);     /* the part of the scene in front of the hole */
    draw_sprites(&front, &progGlow, 0);
}

static void render_frame(void)
{
    ensure_fbo();
    glEnable(GL_BLEND);
    upload(&glow, SPR_FLOATS); upload(&disc, SPR_FLOATS); upload(&front, SPR_FLOATS); upload(&lines, LINE_FLOATS);
    render_layer(2);
    render_layer(1);
    render_layer(0);
    glow.n = disc.n = front.n = lines.n = 0;

    /* combine the layers and tone map to the screen */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, scrW, scrH);
    glDisable(GL_BLEND);
    glUseProgram(progPost);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, fboLow2Tex); glUniform1i(postLow2, 2);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, fboLowTex); glUniform1i(postLow, 1);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, fboTex); glUniform1i(postTex, 0);
    glUniform4f(postFlash, flashR, flashG, flashB, flashA > 0.002f ? clampf(flashA, 0, 1) : 0);
    glBindVertexArray(vaoPost);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

/* ------------------------------------------------------------------ */
/* Shared state: particles and background stars                        */
/* ------------------------------------------------------------------ */
typedef struct { float p[3], v[3], q[3], c[4]; float s, f0, f1, f2, f3; int k; } Part;
#define MAXP 60000
static Part pt[MAXP];
static int np;

#define NSTAR 3500
static float star[NSTAR][7];   /* dir xyz, rgb, magnitude */

static void init_stars(void)
{
    for (int i = 0; i < NSTAR; i++) {
        rand_unit(star[i]);
        float t = frand();      /* temperature tint */
        star[i][3] = t < 0.3f ? 1.0f : t < 0.7f ? 0.9f : 0.7f;
        star[i][4] = t < 0.3f ? 0.75f : t < 0.7f ? 0.9f : 0.82f;
        star[i][5] = t < 0.3f ? 0.55f : t < 0.7f ? 1.0f : 1.0f;
        star[i][6] = powf(frand(), 6.0f);
    }
}

static void draw_stars(float alpha)
{
    if (alpha <= 0) return;
    const float D = 2000;
    for (int i = 0; i < NSTAR; i++) {
        float m = star[i][6];
        emit(&glow, camE[0] + star[i][0]*D, camE[1] + star[i][1]*D, camE[2] + star[i][2]*D,
             star[i][3], star[i][4], star[i][5], alpha * (0.35f + 0.65f * m), D * 0.0022f * (0.7f + 2.2f * m));
    }
}

/* bright star with diffraction spikes */
static void bright_star(float x, float y, float z, float r, float g, float b, float a, float size)
{
    emit(&glow, x, y, z, r, g, b, a * 0.5f, size * 4);
    emit(&glow, x, y, z, 1, 1, 1, a, size);
    for (int i = 1; i <= 6; i++) {
        float d = size * 0.45f * i, aa = a * 0.45f * (1 - i / 7.0f), ss = size * 0.35f;
        for (int s = -1; s <= 1; s += 2) {
            emit(&glow, x + camR[0]*d*s, y + camR[1]*d*s, z + camR[2]*d*s, r, g, b, aa, ss);
            emit(&glow, x + camU[0]*d*s, y + camU[1]*d*s, z + camU[2]*d*s, r, g, b, aa, ss);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scene 0: the Big Bang and the cosmic web                            */
/* ------------------------------------------------------------------ */
#define BB_SEEDS 70
static float bbSeed[BB_SEEDS][3];

static float bb_a(float te) { return te < 0 ? 0 : 55 * (1 - expf(-te / 1.6f)) + 1.6f * te; }

static void bb_init(void)
{
    int nb[BB_SEEDS][3];
    for (int i = 0; i < BB_SEEDS; i++) {
        rand_unit(bbSeed[i]); float r = cbrtf(frand()) * 0.95f;
        for (int k = 0; k < 3; k++) bbSeed[i][k] *= r;
    }
    for (int i = 0; i < BB_SEEDS; i++) {          /* three nearest neighbours */
        float best[3] = { 1e9, 1e9, 1e9 }; nb[i][0] = nb[i][1] = nb[i][2] = i;
        for (int j = 0; j < BB_SEEDS; j++) if (j != i) {
            float dx = bbSeed[i][0]-bbSeed[j][0], dy = bbSeed[i][1]-bbSeed[j][1], dz = bbSeed[i][2]-bbSeed[j][2];
            float d = dx*dx + dy*dy + dz*dz;
            for (int k = 0; k < 3; k++) if (d < best[k]) {
                for (int m = 2; m > k; m--) { best[m] = best[m-1]; nb[i][m] = nb[i][m-1]; }
                best[k] = d; nb[i][k] = j; break;
            }
        }
    }
    np = 46000;
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i];
        rand_unit(p->q); float r = cbrtf(frand());
        for (int k = 0; k < 3; k++) p->q[k] *= r;
        float m = frand(); int s = (int)(frand() * BB_SEEDS) % BB_SEEDS;
        if (m < 0.28f) {                         /* clusters */
            p->k = 0;
            for (int k = 0; k < 3; k++) p->v[k] = bbSeed[s][k] + gauss() * 0.025f;
        } else if (m < 0.85f) {                  /* filaments */
            p->k = 1; int j = nb[s][(int)(frand() * 3) % 3]; float u = frand();
            for (int k = 0; k < 3; k++) p->v[k] = lerp(bbSeed[s][k], bbSeed[j][k], u) + gauss() * 0.012f;
        } else {                                 /* voids */
            p->k = 2;
            for (int k = 0; k < 3; k++) p->v[k] = p->q[k];
        }
        p->f0 = frand() * 4;
    }
}

static void bb_color(float te, float *c)
{
    static const float pal[4][3] = { {1,1,0.95f}, {1,0.85f,0.5f}, {1,0.45f,0.15f}, {0.7f,0.12f,0.06f} };
    float h = clampf((te - 0.5f) / 6, 0, 0.999f) * 3; int i = (int)h; float f = h - i;
    for (int k = 0; k < 3; k++) c[k] = lerp(pal[i][k], pal[i+1][k], f);
}

static void bb_draw(float t)
{
    float te = t - 2.0f, a = bb_a(te);
    float rf = lerp(2.4f, 0.85f, smooth(4, 30, te)), dist = rf * a + 12, ang = 0.6f + 0.035f * t;
    float eye[3] = { dist * cosf(ang), dist * 0.3f, dist * sinf(ang) }, tgt[3] = { 0, 0, 0 };
    set_camera(eye, tgt, 60);

    if (te < 0) {           /* the singularity */
        float b = powf(t / 2, 3);
        emit(&glow, 0, 0, 0, 1, 0.95f, 0.9f, b, 0.4f + 0.6f * b);
        emit(&glow, 0, 0, 0, 0.6f, 0.7f, 1, b * 0.3f, 3 * b);
        return;
    }
    flashA = expf(-te * 2.2f); flashR = 1; flashG = 0.97f; flashB = 0.9f;

    float base[3]; bb_color(te, base);
    float I = 1 - 0.65f * smooth(3, 9, te);
    float sz0 = 0.25f + 0.012f * a;
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i];
        float s = smooth(8 + p->f0, 21 + p->f0, te), c[3], al = 0.35f;
        float x = lerp(p->q[0], p->v[0], s) * a, y = lerp(p->q[1], p->v[1], s) * a, z = lerp(p->q[2], p->v[2], s) * a;
        if (p->k == 2) {
            for (int k = 0; k < 3; k++) c[k] = base[k] * I;
            al *= 1 - 0.85f * s;
        } else {
            static const float sc[3] = { 0.65f, 0.75f, 1.0f };
            for (int k = 0; k < 3; k++) c[k] = lerp(base[k] * I, sc[k], s);
            al = lerp(al, p->k == 0 ? 0.6f : 0.45f, s);
        }
        emit(&glow, x, y, z, c[0], c[1], c[2], al, sz0 * (1 - 0.45f * s));
    }
    float gb = smooth(13, 25, te);
    for (int i = 0; i < BB_SEEDS && gb > 0; i++) {     /* first galaxies light up */
        float h = hash1(i), x = bbSeed[i][0]*a, y = bbSeed[i][1]*a, z = bbSeed[i][2]*a;
        emit(&glow, x, y, z, 0.75f, 0.8f, 1.0f, 0.35f * gb, 0.05f * a * (0.6f + h));
        emit(&glow, x, y, z, 1.0f, 0.95f, 0.85f, 0.8f * gb, 0.012f * a * (0.6f + h));
    }
}

/* ------------------------------------------------------------------ */
/* Scene 1: a spiral galaxy forms                                      */
/* ------------------------------------------------------------------ */
static void gal_init(void)
{
    np = 47000;
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i]; float m = frand(), r;
        if (i < 600) p->k = 2;               /* HII regions */
        else if (i < 4600) p->k = 3;         /* glowing gas along the arms */
        else p->k = m < 0.13f ? 1 : 0;       /* bulge or disk star */
        if (p->k == 1) r = fabsf(gauss()) * 3.2f + 0.2f;
        else { r = 1.5f - 7 * logf(frand() + 1e-6f); if (r > 44) r = 6 + frand() * 38; }
        if (p->k >= 2) r = 5 + frand() * 34;
        p->f0 = r; p->f1 = frand() * 2 * PI;
        p->f2 = gauss() * (p->k == 1 ? 2.2f * (1 - r / 12) : 0.15f + 0.6f * expf(-r / 6));
        rand_unit(p->q); float cr = 55 * cbrtf(frand());
        p->q[0] *= cr; p->q[1] *= cr * 0.6f; p->q[2] *= cr;
        p->f3 = frand();
        float t = frand();
        if (p->k == 1) { p->c[0] = 1; p->c[1] = 0.82f; p->c[2] = 0.55f; p->s = 0.5f; }
        else if (p->k == 2) { p->c[0] = 1; p->c[1] = 0.32f; p->c[2] = 0.5f; p->s = 0.9f + frand(); }
        else if (p->k == 3) { p->c[0] = 0.25f; p->c[1] = 0.38f; p->c[2] = 0.75f; p->s = 4 + 3 * frand(); }
        else if (t < 0.65f) { p->c[0] = 0.7f; p->c[1] = 0.8f; p->c[2] = 1; p->s = 0.35f; }
        else if (t < 0.9f) { p->c[0] = 1; p->c[1] = 0.9f; p->c[2] = 0.7f; p->s = 0.35f; }
        else { p->c[0] = 0.55f; p->c[1] = 0.7f; p->c[2] = 1; p->s = 0.7f; }
        if (p->k == 0 && r < 8) { float w = 1 - r / 8; p->c[2] = lerp(p->c[2], 0.6f, w); p->c[1] = lerp(p->c[1], 0.85f, w); }
    }
}

static void gal_draw(float t)
{
    float u = t / 38, az = 1.2f + 0.06f * t, R = lerp(135, 48, smooth(0, 1, u)), h = 62 * sinf(PI * u) + 8;
    float eye[3] = { R * cosf(az), h, R * sinf(az) }, tgt[3] = { 0, 0, 0 };
    set_camera(eye, tgt, 60);
    draw_stars(0.8f);

    for (int i = 0; i < np; i++) {
        Part *p = &pt[i];
        float r = p->f0, form = smooth(0, 16, t - r * 0.12f - p->f3 * 2);
        float w = 2.5f / (r + 2), th = p->f1 + w * t, tw = (p->k == 1) ? 0 : r * 0.16f;
        float ea = r * cosf(th), eb = r * (p->k == 1 ? 0.92f : 0.72f) * sinf(th);
        float dx = ea * cosf(tw) - eb * sinf(tw), dz = ea * sinf(tw) + eb * cosf(tw), dy = p->f2;
        float cx = p->q[0], cy = p->q[1], cz = p->q[2]; roty(&cx, &cz, 0.03f * t);
        float x = lerp(cx, dx, form), y = lerp(cy, dy, form), z = lerp(cz, dz, form);
        roty(&x, &z, (1 - form) * 1.8f);
        float gc[3] = { 0.35f, 0.3f, 0.65f }, al, sz;
        if (p->k == 3) { al = lerp(0.03f, 0.07f, form); sz = p->s; }
        else if (p->k == 2) { al = 0.55f * form * (0.7f + 0.3f * sinf(t * 0.7f + i)); sz = p->s; }
        else { al = lerp(0.1f, p->k == 1 ? 0.3f : 0.5f, form); sz = lerp(1.6f, p->s, form); }
        emit(&glow, x, y, z, lerp(gc[0], p->c[0], form), lerp(gc[1], p->c[1], form), lerp(gc[2], p->c[2], form), al, sz);
    }
    float fc = smooth(2, 16, t);
    emit(&glow, 0, 0, 0, 1, 0.75f, 0.5f, 0.3f * fc, 16);
    emit(&glow, 0, 0, 0, 1, 0.9f, 0.75f, 0.5f * fc, 4);
}

/* ------------------------------------------------------------------ */
/* Scene 2: galaxy collision (restricted N-body)                       */
/* ------------------------------------------------------------------ */
static float core[2][6], coreM[2] = { 1.0f, 0.8f }, colSimT;
#define COL_EPS2 1.0f

static void col_accel(const float *x, float *a)
{
    a[0] = a[1] = a[2] = 0;
    for (int j = 0; j < 2; j++) {
        float dx = core[j][0]-x[0], dy = core[j][1]-x[1], dz = core[j][2]-x[2];
        float d2 = dx*dx + dy*dy + dz*dz + COL_EPS2, inv = coreM[j] / (d2 * sqrtf(d2));
        a[0] += dx * inv; a[1] += dy * inv; a[2] += dz * inv;
    }
}

static void col_init(void)
{
    float Mt = coreM[0] + coreM[1], d = 40, q = 6;
    float v = sqrtf(2 * Mt / d), hh = sqrtf(2 * Mt * q), vt = hh / d, vr = -sqrtf(v*v - vt*vt);
    float Rr[3] = { d, 0, 0 }, Vr[3] = { vr, 0, vt };
    for (int k = 0; k < 3; k++) {
        core[0][k] = -coreM[1] / Mt * Rr[k]; core[0][k+3] = -coreM[1] / Mt * Vr[k];
        core[1][k] =  coreM[0] / Mt * Rr[k]; core[1][k+3] =  coreM[0] / Mt * Vr[k];
    }
    float n[2][3] = { { 0, -cosf(0.35f), -sinf(0.35f) }, { 0.4f, -0.7f, 0.6f } };
    np = 26000;
    for (int g = 0; g < 2; g++) {
        float *nn = n[g], e1[3], e2[3], ax[3] = { 1, 0, 0 };
        norm3(nn);
        if (fabsf(nn[0]) > 0.9f) { ax[0] = 0; ax[1] = 1; }
        cross3(nn, ax, e1); norm3(e1); cross3(nn, e1, e2);
        for (int i = g * np / 2; i < (g + 1) * np / 2; i++) {
            Part *p = &pt[i];
            float r = 1.0f + 11 * powf(frand(), 0.8f), ph = frand() * 2 * PI, cp = cosf(ph), sp = sinf(ph);
            float vc = sqrtf(coreM[g] * r * r / powf(r * r + COL_EPS2, 1.5f)), hz = gauss() * 0.15f;
            for (int k = 0; k < 3; k++) {
                p->p[k] = core[g][k] + r * (cp * e1[k] + sp * e2[k]) + hz * nn[k];
                p->v[k] = core[g][k+3] + vc * (cp * e2[k] - sp * e1[k]);
            }
            p->k = g;
            float w = smooth(5, 0, r);
            if (g == 0) { p->c[0] = lerp(0.62f, 1, w); p->c[1] = lerp(0.74f, 0.88f, w); p->c[2] = lerp(1, 0.65f, w); }
            else        { p->c[0] = 1; p->c[1] = lerp(0.78f, 0.9f, w); p->c[2] = lerp(0.55f, 0.7f, w); }
            p->s = 0.28f + 0.25f * powf(frand(), 8);
        }
    }
    colSimT = 0;
}

static void col_step(float h)
{
    /* cores: mutual gravity + a dynamical friction stand-in so they merge */
    float rel[3], vrel[3], d2 = COL_EPS2;
    for (int k = 0; k < 3; k++) { rel[k] = core[1][k] - core[0][k]; vrel[k] = core[1][k+3] - core[0][k+3]; d2 += rel[k]*rel[k]; }
    float d = sqrtf(d2), inv = 1 / (d2 * d), fr = 0.09f * expf(-d / 9), Mt = coreM[0] + coreM[1];
    for (int k = 0; k < 3; k++) {
        float arel = -Mt * rel[k] * inv - fr * vrel[k];
        core[0][k+3] -= coreM[1] / Mt * arel * h;
        core[1][k+3] += coreM[0] / Mt * arel * h;
    }
    for (int j = 0; j < 2; j++) for (int k = 0; k < 3; k++) core[j][k] += core[j][k+3] * h;
    for (int i = 0; i < np; i++) {          /* semi-implicit Euler for test stars */
        Part *p = &pt[i]; float a[3];
        col_accel(p->p, a);
        for (int k = 0; k < 3; k++) { p->v[k] += a[k] * h; p->p[k] += p->v[k] * h; }
    }
}

static void col_update(float t, float dt)
{
    float target = t * 5.5f;
    while (colSimT < target) { col_step(0.05f); colSimT += 0.05f; }
}

static void col_draw(float t)
{
    float az = 0.4f + 0.022f * t, R = lerp(46, 75, smooth(8, 40, t)), h = lerp(26, 16, smooth(0, 40, t));
    float c[3]; for (int k = 0; k < 3; k++) c[k] = (core[0][k] * coreM[0] + core[1][k] * coreM[1]) / (coreM[0] + coreM[1]);
    float eye[3] = { c[0] + R * cosf(az), c[1] + h, c[2] + R * sinf(az) };
    set_camera(eye, c, 60);
    draw_stars(0.8f);
    float dx = core[0][0]-core[1][0], dy = core[0][1]-core[1][1], dz = core[0][2]-core[1][2];
    float burst = smooth(14, 4, sqrtf(dx*dx + dy*dy + dz*dz)) * smooth(1, 15, t);
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i];
        emit(&glow, p->p[0], p->p[1], p->p[2], p->c[0], p->c[1], p->c[2], 0.6f, p->s);
    }
    for (int j = 0; j < 2; j++) {
        emit(&glow, core[j][0], core[j][1], core[j][2], 1, 0.85f, 0.6f, 0.45f + 0.3f * burst, 6 + 4 * burst);
        emit(&glow, core[j][0], core[j][1], core[j][2], 1, 0.95f, 0.85f, 0.9f, 1.6f);
    }
    /* starburst: pink star-forming knots near the interacting cores */
    for (int i = 0; i < 300 && burst > 0.01f; i++) {
        Part *p = &pt[(i * 97) % np];
        emit(&glow, p->p[0], p->p[1], p->p[2], 1, 0.35f, 0.55f, 0.5f * burst, 1.0f);
    }
}

/* ------------------------------------------------------------------ */
/* Scene: supermassive black hole with a tidal disruption event       */
/* ------------------------------------------------------------------ */
#define BH_GM 60.0f
#define BH_NDISK 30000
#define BH_NSTAR 3500
#define BH_NBAND 3000
static float bhFlare, bhSimT, bhStarC[3];
static float band[BH_NBAND][7];

static void bh_init(void)
{
    np = BH_NDISK + BH_NSTAR;
    for (int i = 0; i < BH_NDISK; i++) {             /* accretion disk */
        Part *p = &pt[i];
        p->f0 = 3 + 13 * powf(frand(), 0.9f); p->f1 = frand() * 2 * PI; p->f2 = gauss() * 0.05f * sqrtf(p->f0);
        p->s = 0.25f + 0.35f * frand(); p->k = 0;
    }
    /* a sun-like star on an eccentric plunge (pericentre inside its tidal radius) */
    float d = 24, q = 4.2f, Q = 30, sa = (q + Q) / 2, ec = (Q - q) / (Q + q);
    float v = sqrtf(BH_GM * (2 / d - 1 / sa)), vt = sqrtf(BH_GM * sa * (1 - ec * ec)) / d, vr = -sqrtf(v * v - vt * vt);
    float e1[3] = { 0.8f, 0.25f, -0.55f }, e2[3], nrm[3] = { 0.25f, 1, 0.3f };
    norm3(e1); cross3(nrm, e1, e2); norm3(e2);
    for (int i = BH_NDISK; i < np; i++) {
        Part *p = &pt[i]; float o[3]; rand_unit(o); float rr = 1.3f * cbrtf(frand());
        for (int k = 0; k < 3; k++) { p->p[k] = d * e1[k] + o[k] * rr; p->v[k] = vr * e1[k] + vt * e2[k] + o[k] * 0.02f; }
        p->k = 1; p->s = 0.3f + 0.2f * frand();
    }
    for (int k = 0; k < 3; k++) bhStarC[k] = d * e1[k];
    /* a faint band of distant galaxy light so the lensing distortion shows */
    float bn[3] = { 0.2f, 0.9f, -0.4f }, b1[3], b2[3], ax[3] = { 1, 0, 0 };
    norm3(bn); cross3(bn, ax, b1); norm3(b1); cross3(bn, b1, b2);
    for (int i = 0; i < BH_NBAND; i++) {
        float a = frand() * 2 * PI, w = gauss() * 0.12f;
        for (int k = 0; k < 3; k++) band[i][k] = 1500 * (cosf(a) * b1[k] + sinf(a) * b2[k] + w * bn[k]);
        float t = frand();
        band[i][3] = 0.9f; band[i][4] = t < 0.5f ? 0.75f : 0.85f; band[i][5] = t < 0.5f ? 0.6f : 0.95f;
        band[i][6] = 18 + 30 * frand();
    }
    bhFlare = 0; bhSimT = 0;
}

static void bh_update(float t, float dt)
{
    /* disk: Keplerian rotation with slow inflow; respawn after plunging */
    for (int i = 0; i < BH_NDISK; i++) {
        Part *p = &pt[i]; float r = p->f0;
        p->f1 += sqrtf(BH_GM / (r * r * r)) * dt;
        p->f0 -= dt * (0.04f + (r < 3.4f ? 1.5f : 0));
        if (p->f0 < 2.4f) { p->f0 = 3 + 13 * frand(); p->f1 = frand() * 2 * PI; }
    }
    if (t < 4) return;                              /* the star enters at t = 4 */
    float target = t - 4;
    while (bhSimT < target) {
        const float h = 0.004f;
        for (int i = BH_NDISK; i < np; i++) {
            Part *p = &pt[i]; if (p->k != 1) continue;
            float r2 = p->p[0]*p->p[0] + p->p[1]*p->p[1] + p->p[2]*p->p[2] + 0.05f, r = sqrtf(r2);
            if (r < 2.6f) { p->k = -1; bhFlare += 0.0015f; continue; }    /* swallowed */
            float inv = -BH_GM / (r2 * r);
            /* self-gravity of the star holds it together until the tidal field wins */
            float sx = bhStarC[0] - p->p[0], sy = bhStarC[1] - p->p[1], sz = bhStarC[2] - p->p[2];
            float s2 = sx*sx + sy*sy + sz*sz + 0.08f, sg = r > 13 ? 1.2f / (s2 * sqrtf(s2)) : 0;
            p->v[0] += (p->p[0] * inv + sx * sg) * h; p->v[1] += (p->p[1] * inv + sy * sg) * h; p->v[2] += (p->p[2] * inv + sz * sg) * h;
            p->p[0] += p->v[0] * h; p->p[1] += p->v[1] * h; p->p[2] += p->v[2] * h;
        }
        bhSimT += h;
        if (((int)(bhSimT / h)) % 25 == 0) {        /* refresh the star's centre */
            float c[3] = { 0, 0, 0 }; int n = 0;
            for (int i = BH_NDISK; i < np; i++) if (pt[i].k == 1) { for (int k = 0; k < 3; k++) c[k] += pt[i].p[k]; n++; }
            if (n) for (int k = 0; k < 3; k++) bhStarC[k] = c[k] / n;
        }
    }
    bhFlare *= expf(-dt / 7);
}

static void bh_draw(float t)
{
    float dist = lerp(95, 32, smooth(0, 20, t)) + 6 * sinf(t * 0.15f), az = 0.4f + 0.035f * t;
    float el = lerp(0.45f, 0.11f, smooth(0, 22, t));
    float eye[3] = { dist * cosf(el) * cosf(az), dist * sinf(el), dist * cosf(el) * sinf(az) }, tgt[3] = { 0, 0, 0 };
    set_camera(eye, tgt, 55);
    lens_setup(tgt, 1.0f);
    draw_stars(0.9f);
    for (int i = 0; i < BH_NBAND; i++)
        emit(&glow, band[i][0], band[i][1], band[i][2], band[i][3], band[i][4], band[i][5], 0.05f, band[i][6]);

    float fl = clampf(bhFlare, 0, 2.0f), toC[3];
    /* accretion disk with relativistic beaming */
    for (int i = 0; i < BH_NDISK; i++) {
        Part *p = &pt[i]; float r = p->f0, c = cosf(p->f1), s = sinf(p->f1);
        float x = r * c, y = p->f2, z = r * s;
        toC[0] = camE[0] - x; toC[1] = camE[1] - y; toC[2] = camE[2] - z; norm3(toC);
        float beta = 0.55f * sqrtf(3 / r), cosang = -s * toC[0] + c * toC[2];
        float D = 1 / (sqrtf(1 / (1 - beta * beta)) * (1 - beta * cosang)), boost = clampf(D * D * D, 0.15f, 4);
        float hot = smooth(12, 3, r), plunge = smooth(2.4f, 3.4f, r);
        float cr = 1, cg = lerp(0.35f, 0.9f, hot), cb = lerp(0.1f, 0.75f, hot);
        cb = clampf(cb * lerp(1, D, 0.5f), 0, 1);
        float al = 0.1f * boost * plunge * (0.8f + 0.35f * fl) * smooth(16.5f, 13, r);
        emit(&glow, x, y, z, cr, cg, cb, al, p->s * (1 + r * 0.06f));
    }
    /* the disrupted star and its debris stream */
    float intact = 1;
    { float cr2 = bhStarC[0]*bhStarC[0] + bhStarC[1]*bhStarC[1] + bhStarC[2]*bhStarC[2]; intact = t < 4 ? 0 : smooth(9, 16, sqrtf(cr2)); }
    for (int i = BH_NDISK; i < np && t >= 4; i++) {
        Part *p = &pt[i]; if (p->k != 1) continue;
        float r = sqrtf(p->p[0]*p->p[0] + p->p[1]*p->p[1] + p->p[2]*p->p[2]), heat = smooth(10, 3, r);
        emit(&glow, p->p[0], p->p[1], p->p[2], 1, lerp(0.75f, 0.95f, heat), lerp(0.45f, 0.9f, heat),
                  (0.3f + 0.3f * heat) * (1 - 0.6f * intact) * smooth(4, 5, t), p->s);
    }
    if (intact > 0.01f && t >= 4) {
        float a = intact * smooth(4, 5.5f, t);
        emit(&glow, bhStarC[0], bhStarC[1], bhStarC[2], 1, 0.85f, 0.55f, 0.5f * a, 5);
        emit(&glow, bhStarC[0], bhStarC[1], bhStarC[2], 1, 0.97f, 0.9f, 0.9f * a, 1.6f);
    }
    /* polar jets, brightened by the feeding frenzy */
    float jp = 0.35f + 0.5f * fl;
    for (int i = 0; i < 900; i++) {
        float ph = fmodf(i * 0.618034f + t * 0.55f, 1.0f), sgn = (i & 1) ? 1 : -1, hgt = sgn * (2.5f + ph * 75);
        float spread = 0.25f + ph * 3.5f, ang = ph * 14 + t * 1.5f + hash1(i) * 6.28f, rr = spread * (0.3f + 0.7f * hash1(i + 7));
        emit(&glow, rr * cosf(ang), hgt, rr * sinf(ang), 0.55f, 0.72f, 1, jp * 0.3f * (1 - ph), 0.6f + ph * 3);
    }
    /* the shadow, plus a thin photon ring at its edge */
    emit(&disc, 0, 0, 0, 0, 0, 0, 1, 2 * lensShadow * 0.98f);
    for (int i = 0; i < 260; i++) {
        float a = i * 2 * PI / 260, rr = lensShadow * 1.04f;
        float x = (camR[0] * cosf(a) + camU[0] * sinf(a)) * rr, y = (camR[1] * cosf(a) + camU[1] * sinf(a)) * rr,
              z = (camR[2] * cosf(a) + camU[2] * sinf(a)) * rr;
        emit_nl(&glow, x, y, z, 1, 0.75f, 0.45f, 0.3f * (0.8f + 0.4f * fl) * (0.1f + 0.9f * smooth(-0.35f, 0.05f, sinf(a))), 0.3f);
    }
    if (fl > 0.05f) emit(&front, 0, 0, 0, 1, 0.9f, 0.8f, 0.08f * fl, 30);
}

/* ------------------------------------------------------------------ */
/* Scene: stellar nursery                                              */
/* ------------------------------------------------------------------ */
#define NSEED 28
static float seed[NSEED][8];  /* pos xyz, ignite time, rgb, size */

static void neb_init(void)
{
    float blob[10][4];
    for (int i = 0; i < 10; i++) {
        blob[i][0] = -50 + 11 * i; blob[i][1] = 12 * sinf(i * 0.7f); blob[i][2] = 8 * cosf(i * 1.1f);
        blob[i][3] = 6 + 5 * frand();
    }
    np = 15000;
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i]; int b = (int)(frand() * 10) % 10;
        for (int k = 0; k < 3; k++) p->p[k] = blob[b][k] + gauss() * blob[b][3] * (k == 1 ? 0.7f : 1.0f);
        float m = frand();
        if (i < 9000) {
            p->k = 0;
            if (m < 0.5f) { p->c[0] = 0.6f; p->c[1] = 0.13f; p->c[2] = 0.1f; }
            else if (m < 0.75f) { p->c[0] = 0.6f; p->c[1] = 0.3f; p->c[2] = 0.1f; }
            else { p->c[0] = 0.1f; p->c[1] = 0.36f; p->c[2] = 0.42f; }
            p->s = 3 + 5 * frand();
        } else { p->k = 1; p->c[0] = 0.45f; p->c[1] = 0.3f; p->c[2] = 0.2f; p->s = 0.35f + 0.3f * frand(); }
        p->f0 = 0;
    }
    for (int i = 0; i < NSEED; i++) {
        memcpy(seed[i], pt[(int)(frand() * 9000)].p, sizeof(float) * 3);
        seed[i][3] = 4 + 24 * frand();
        float m = frand();
        if (m < 0.35f) { seed[i][4] = 0.65f; seed[i][5] = 0.8f; seed[i][6] = 1; seed[i][7] = 1.4f; }
        else if (m < 0.75f) { seed[i][4] = 1; seed[i][5] = 0.95f; seed[i][6] = 0.85f; seed[i][7] = 1.0f; }
        else { seed[i][4] = 1; seed[i][5] = 0.7f; seed[i][6] = 0.45f; seed[i][7] = 0.8f; }
    }
    for (int i = 0; i < np; i++) {      /* each gas parcel belongs to the nearest seed */
        Part *p = &pt[i]; float best = 14 * 14; p->v[0] = -1;
        for (int j = 0; j < NSEED; j++) {
            float dx = seed[j][0]-p->p[0], dy = seed[j][1]-p->p[1], dz = seed[j][2]-p->p[2], d = dx*dx + dy*dy + dz*dz;
            if (d < best) { best = d; p->v[0] = j; }
        }
    }
}

static void neb_update(float t, float dt)
{
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i]; float *x = p->p;
        x[0] += 0.35f * sinf(x[1] * 0.08f + t * 0.1f) * dt;
        x[1] += 0.25f * sinf(x[2] * 0.09f + t * 0.13f) * dt;
        x[2] += 0.35f * sinf(x[0] * 0.07f + t * 0.11f) * dt;
        int j = (int)p->v[0];
        if (j < 0) continue;
        float dx = seed[j][0]-x[0], dy = seed[j][1]-x[1], dz = seed[j][2]-x[2], d = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-3f;
        float ti = seed[j][3], sp;
        if (t < ti) sp = d > 1.5f ? 0.7f * smooth(ti - 14, ti, t) : 0;     /* gravitational collapse */
        else { sp = -6 * expf(-(t - ti) / 5) * smooth(18, 2, d); p->f0 = clampf(p->f0 + dt * 0.12f * smooth(16, 3, d), 0, 1); }
        x[0] += dx / d * sp * dt; x[1] += dy / d * sp * dt; x[2] += dz / d * sp * dt;
    }
}

static void neb_draw(float t)
{
    float u = t / 36;
    float eye[3] = { lerp(-75, 50, u), 6 + 4 * sinf(u * 3), 22 - 8 * u }, tgt[3] = { eye[0] + 30, 0, 0 };
    set_camera(eye, tgt, 62);
    draw_stars(0.6f);
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i]; float io = p->f0, br = 1;
        int j = (int)p->v[0];
        if (j >= 0) {
            float ti = seed[j][3], pre = smooth(ti - 8, ti, t) * (t < ti + 1 ? 1 : 0);
            br += pre * 0.6f;
        }
        float r = lerp(p->c[0], 0.25f, io) * br, g = lerp(p->c[1], 0.55f, io) * br, b = lerp(p->c[2], 0.85f, io) * br;
        emit(&glow, p->p[0], p->p[1], p->p[2], r, g, b, p->k ? 0.25f : 0.075f + 0.04f * io, p->s);
    }
    for (int j = 0; j < NSEED; j++) {
        float *s = seed[j], ti = s[3];
        if (t < ti) {
            float pr = smooth(ti - 9, ti, t);
            emit(&glow, s[0], s[1], s[2], 0.9f, 0.3f, 0.1f, 0.45f * pr, 1 + 3 * pr);
        } else {
            float e = t - ti;
            if (e < 3) emit(&glow, s[0], s[1], s[2], 1, 0.95f, 0.9f, expf(-e * 1.5f), 6 + 24 * e);
            bright_star(s[0], s[1], s[2], s[4], s[5], s[6], 0.9f * smooth(0, 1, e), s[7]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scene 4: protoplanetary disk -> planetary system                    */
/* ------------------------------------------------------------------ */
#define NPL 7
static const float plR[NPL] = { 6, 9.5f, 13.5f, 26, 37, 49, 59 };
static const float plW[NPL] = { 1.5f, 2.0f, 2.6f, 6.5f, 6.5f, 5.0f, 4.0f };
static const float plS[NPL] = { 0.55f, 0.9f, 0.8f, 3.4f, 2.8f, 1.7f, 1.6f };
static const float plC[NPL][3] = { {0.7f,0.6f,0.55f}, {0.95f,0.8f,0.55f}, {0.4f,0.6f,1.0f}, {0.95f,0.75f,0.5f},
                                   {0.95f,0.88f,0.65f}, {0.55f,0.85f,0.95f}, {0.35f,0.5f,1.0f} };
static float plPh[NPL];

static float kep(float r) { return 20 / powf(r, 1.5f); }

static void sol_init(void)
{
    np = 34000;
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i]; float r;
        if (i < 2500) { p->k = 1; r = 17 + 4 * frand(); }          /* asteroid belt */
        else if (i < 5000) { p->k = 2; r = 65 + 12 * frand(); }     /* comet belt */
        else { p->k = 0; r = 2.5f + 62 * powf(frand(), 0.85f); }
        p->f0 = r; p->f1 = frand() * 2 * PI; p->f2 = gauss() * 0.045f * r; p->f3 = 1e9f;
        p->v[0] = frand(); p->v[1] = -1;
        rand_unit(p->q); float cr = 75 * cbrtf(frand()); for (int k = 0; k < 3; k++) p->q[k] *= cr;
        float hot = smooth(25, 3, r);
        if (r > 30 && frand() < 0.4f) { p->c[0] = 0.6f; p->c[1] = 0.7f; p->c[2] = 0.85f; }   /* ices */
        else { p->c[0] = lerp(0.55f, 1, hot); p->c[1] = lerp(0.38f, 0.65f, hot); p->c[2] = lerp(0.3f, 0.35f, hot); }
        p->s = 0.45f + 0.5f * frand();
        if (p->k) { p->c[0] = 0.6f; p->c[1] = 0.55f; p->c[2] = 0.5f; p->s = 0.3f; }
    }
    for (int j = 0; j < NPL; j++) plPh[j] = frand() * 2 * PI;
}

static void planet_pos(int j, float t, float *o)
{
    float th = plPh[j] + kep(plR[j]) * t;
    o[0] = plR[j] * cosf(th); o[1] = 0; o[2] = plR[j] * sinf(th);
}

static void sol_update(float t, float dt)
{
    if (t < 12) return;
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i];
        if (p->k || p->f3 < 1e8f) continue;
        for (int j = 0; j < NPL; j++) {
            if (fabsf(p->f0 - plR[j]) < plW[j] * smooth(12, 32, t) * p->v[0]) { p->f3 = t; p->v[1] = j; break; }
        }
    }
}

static void sol_draw(float t)
{
    float pp[NPL][3];
    for (int j = 0; j < NPL; j++) planet_pos(j, t, pp[j]);
    /* camera: overview, then follow the giant planet */
    float az = 0.3f + 0.025f * t, R = lerp(155, 105, smooth(0, 30, t)), h = lerp(75, 38, smooth(0, 30, t));
    float eye[3] = { R * cosf(az), h, R * sinf(az) }, tgt[3] = { 0, 0, 0 };
    float b = smooth(31, 42, t);
    if (b > 0) {
        /* trail the giant so the star is ~100 degrees from it: a half-lit planet with the sun in frame */
        int J = 3; float th = plPh[J] + kep(plR[J]) * t, ur[2] = { cosf(th), sinf(th) }, ut[2] = { -sinf(th), cosf(th) };
        float fe[3] = { pp[J][0] + 18 * (0.17f * ur[0] - 0.98f * ut[0]), 3.0f, pp[J][2] + 18 * (0.17f * ur[1] - 0.98f * ut[1]) };
        float ft[3] = { pp[J][0] * 0.6f, 0, pp[J][2] * 0.6f };
        for (int k = 0; k < 3; k++) { eye[k] = lerp(eye[k], fe[k], b); tgt[k] = lerp(tgt[k], ft[k], b); }
    }
    set_camera(eye, tgt, 60);
    draw_stars(0.7f);

    float evap = 1 - smooth(25, 40, t);
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i]; float r = p->f0;
        float form = smooth(0, 10, t - r * 0.04f);
        float th = p->f1 + kep(r) * t;
        float x = lerp(p->q[0], r * cosf(th), form), y = lerp(p->q[1], p->f2, form), z = lerp(p->q[2], r * sinf(th), form);
        roty(&x, &z, (1 - form) * 2.0f);
        float al = p->k ? lerp(0.2f, 0.5f, form) : lerp(0.22f, 0.3f, form) * evap;
        if (p->f3 < 1e8f) {                      /* being swept up by a planet */
            float e = smooth(0, 2, t - p->f3); int j = (int)p->v[1];
            if (e >= 1) continue;
            x = lerp(x, pp[j][0], e); y = lerp(y, pp[j][1], e); z = lerp(z, pp[j][2], e);
            al *= 1 - e * 0.5f;
        }
        emit(&glow, x, y, z, p->c[0], p->c[1], p->c[2], al, p->s);
    }
    /* bipolar jets of the young star */
    float js = smooth(2, 4, t) * smooth(15, 10, t);
    for (int i = 0; i < 700 && js > 0; i++) {
        float ph = fmodf(i * 0.618034f + t * 0.45f, 1.0f), hgt = ph * 50 * (i & 1 ? 1 : -1);
        float sp = 0.4f + ph * 2.5f;
        emit(&glow, (hash1(i) - 0.5f) * sp, hgt, (hash1(i + 999) - 0.5f) * sp, 0.5f, 0.7f, 1, js * 0.35f * (1 - ph), 1.2f + ph * 2);
    }
    /* the star */
    float ign = smooth(8, 11, t);
    float sc[3] = { lerp(1, 1, ign), lerp(0.4f, 0.93f, ign), lerp(0.15f, 0.75f, ign) };
    emit(&glow, 0, 0, 0, sc[0], sc[1], sc[2], 0.5f + 0.15f * ign, 14 + 6 * ign);
    emit(&glow, 0, 0, 0, 1, 1, 0.95f, 0.5f + 0.5f * ign, 2 + 2 * ign);
    if (t > 9 && t < 13) emit(&glow, 0, 0, 0, 1, 0.95f, 0.85f, expf(-(t - 9) * 1.2f), 10 + 60 * (t - 9));

    /* planets */
    float pg = smooth(10, 32, t);
    for (int j = 0; j < NPL && pg > 0; j++) {
        float sz = 0.2f + plS[j] * pg;
        emit(&disc, pp[j][0], pp[j][1], pp[j][2], plC[j][0], plC[j][1], plC[j][2], smooth(10, 14, t), sz);
        emit(&glow, pp[j][0], pp[j][1], pp[j][2], plC[j][0], plC[j][1], plC[j][2], 0.15f, sz * 2.5f);
        if (j == 4) for (int k = 0; k < 160; k++) {     /* a ring system */
            float a = k * 2 * PI / 160, rr = sz * (1.25f + 0.35f * hash1(k));
            float rx = rr * cosf(a), ry = rr * sinf(a) * 0.35f, rz = rr * sinf(a) * 0.94f;
            emit(&glow, pp[j][0] + rx, pp[j][1] + ry, pp[j][2] + rz, 0.9f, 0.85f, 0.7f, 0.25f * pg, sz * 0.18f);
        }
    }
    /* orbit lines */
    float ol = smooth(30, 35, t) * 0.25f;
    for (int j = 0; j < NPL && ol > 0; j++) for (int k = 0; k < 128; k++) {
        float a0 = k * 2 * PI / 128, a1 = (k + 1) * 2 * PI / 128;
        line(plR[j]*cosf(a0), 0, plR[j]*sinf(a0), plR[j]*cosf(a1), 0, plR[j]*sinf(a1), 0.5f, 0.6f, 0.9f, ol);
    }
}

/* ------------------------------------------------------------------ */
/* Scene 5: supernova and pulsar                                       */
/* ------------------------------------------------------------------ */
#define SN_T 12.6f
#define NFING 160
static float fing[NFING][4];

static void sn_init(void)
{
    for (int i = 0; i < NFING; i++) { rand_unit(fing[i]); fing[i][3] = 0.8f + 0.4f * frand(); }
    np = 38000;
    for (int i = 0; i < np; i++) {
        Part *p = &pt[i];
        if (i < 7000) { p->k = 0; rand_unit(p->q); p->s = 2.2f + frand(); continue; }   /* stellar surface */
        p->k = 1; int f = (int)(frand() * NFING) % NFING;
        for (int k = 0; k < 3; k++) p->q[k] = fing[f][k] + gauss() * 0.045f;
        norm3(p->q);
        p->f0 = frand() < 0.7f ? fing[f][3] * (0.55f + 0.45f * sqrtf(frand())) : (0.8f + 0.1f * gauss());
        float m = frand();
        if (m < 0.45f) { p->c[0] = 1; p->c[1] = 0.25f; p->c[2] = 0.2f; }
        else if (m < 0.75f) { p->c[0] = 0.2f; p->c[1] = 0.9f; p->c[2] = 0.8f; }
        else if (m < 0.9f) { p->c[0] = 0.7f; p->c[1] = 0.1f; p->c[2] = 0.2f; }
        else { p->c[0] = 0.5f; p->c[1] = 0.6f; p->c[2] = 1; }
        p->s = 0.6f + 0.9f * frand();
    }
}

static void sn_draw(float t)
{
    float te = t - SN_T, az = 0.05f * t, R = te < 0 ? lerp(75, 55, smooth(0, 12, t)) : lerp(55, 170, smooth(0, 20, te));
    float eye[3] = { R * cosf(az), 0.3f * R, R * sinf(az) }, tgt[3] = { 0, 0, 0 };
    set_camera(eye, tgt, 60);
    draw_stars(0.8f);

    if (te < 0) {           /* the red supergiant, then core collapse */
        float col = smooth(11.6f, 12.6f, t), Rs = 13 * (1 - 0.88f * col) * (1 + 0.03f * sinf(t * 0.9f));
        float hot = col;
        for (int i = 0; i < 7000; i++) {
            Part *p = &pt[i]; float *d = p->q;
            float n = 1 + 0.07f * sinf(3 * d[0] + t * 0.7f) * sinf(4 * d[1] - t * 0.5f) + 0.04f * sinf(7 * d[2] + t);
            emit(&glow, d[0] * Rs * n, d[1] * Rs * n, d[2] * Rs * n, 1, lerp(0.35f, 0.8f, hot), lerp(0.1f, 0.9f, hot), 0.13f, p->s * (1 - 0.7f * col));
        }
        emit(&glow, 0, 0, 0, 1, lerp(0.4f, 0.85f, hot), lerp(0.15f, 1, hot), 0.4f, Rs * 3.5f);
        emit(&glow, 0, 0, 0, 1, 0.8f, 0.6f, 0.3f + 0.7f * hot, Rs * 1.2f);
        return;
    }
    flashA = expf(-te * 2.0f) * 0.95f; flashR = 0.9f; flashG = 0.95f; flashB = 1;
    float rr = 62 * (1 - expf(-te / 5)), cool = smooth(0, 5, te), fade = 0.35f + 0.65f * expf(-te / 12);
    for (int i = 7000; i < np; i++) {
        Part *p = &pt[i]; float rad = rr * p->f0;
        emit(&glow, p->q[0] * rad, p->q[1] * rad, p->q[2] * rad,
             lerp(1, p->c[0], cool), lerp(1, p->c[1], cool), lerp(1, p->c[2], cool),
             0.4f * fade, p->s * (0.4f + 0.6f * cool) * (1 + rad / 80));
    }
    /* pulsar with sweeping beams */
    float pv = smooth(1.5f, 4, te);
    bright_star(0, 0, 0, 0.6f, 0.8f, 1, pv, 2.5f);
    float ax[3] = { 0.3f, 1, 0.1f }, e1[3], e2[3], up[3] = { 1, 0, 0 };
    norm3(ax); cross3(ax, up, e1); norm3(e1); cross3(ax, e1, e2);
    float w = te * 2 * PI / 1.4f, sa = sinf(0.7f), ca = cosf(0.7f);
    float bd[3]; for (int k = 0; k < 3; k++) bd[k] = (cosf(w) * e1[k] + sinf(w) * e2[k]) * sa + ax[k] * ca;
    for (int i = 1; i <= 45 && pv > 0; i++) for (int s = -1; s <= 1; s += 2) {
        float d = i * 0.8f * s;
        emit(&glow, bd[0] * d, bd[1] * d, bd[2] * d, 0.55f, 0.75f, 1, 0.5f * pv * (1 - i / 46.0f), 1.2f + i * 0.08f);
    }
}

/* ------------------------------------------------------------------ */
/* Scene table                                                         */
/* ------------------------------------------------------------------ */
typedef struct { void (*init)(void); void (*update)(float, float); void (*draw)(float); float dur; } Scene;
static Scene scenes[] = {
    { bb_init,  NULL,       bb_draw,  36 },
    { gal_init, NULL,       gal_draw, 38 },
    { col_init, col_update, col_draw, 40 },
    { bh_init,  bh_update,  bh_draw,  42 },
    { neb_init, neb_update, neb_draw, 36 },
    { sol_init, sol_update, sol_draw, 46 },
    { sn_init,  NULL,       sn_draw,  36 },
};
#define NSCENES ((int)(sizeof scenes / sizeof scenes[0]))
static const char *sceneName[NSCENES] = { "bigbang", "galaxy", "collision", "blackhole", "nursery", "planets", "supernova" };
static int enabled[NSCENES] = { 1, 1, 1, 1, 1, 1, 1 };
static int shuffle = 0;
static int cur = 0;
static float st = 0;

/* scene order: either the story order, or a shuffled deck of the enabled scenes */
static int deck[NSCENES], deckN = 0, deckPos = 0;

static void start_scene(int i) { cur = (i + NSCENES) % NSCENES; st = 0; scenes[cur].init(); }

static void deal(void)
{
    deckN = 0;
    for (int i = 0; i < NSCENES; i++) if (enabled[i]) deck[deckN++] = i;
    for (int i = deckN - 1; i > 0; i--) { int j = (int)(frand() * (i + 1)) % (i + 1), t = deck[i]; deck[i] = deck[j]; deck[j] = t; }
    if (deckN > 1 && deck[0] == cur) { int t = deck[0]; deck[0] = deck[deckN - 1]; deck[deckN - 1] = t; }   /* no repeats across decks */
    deckPos = 0;
}

static void step_scene(int dir)
{
    if (shuffle && dir > 0) {
        if (deckPos >= deckN) deal();
        start_scene(deck[deckPos++]);
        return;
    }
    int i = cur;
    do i = (i + dir + NSCENES) % NSCENES; while (!enabled[i]);
    start_scene(i);
}

static void advance(float dt)
{
    st += dt;
    if (st >= scenes[cur].dur) step_scene(1);
    if (scenes[cur].update) scenes[cur].update(st, dt);
}

static void draw_scene(void)
{
    float d = scenes[cur].dur;
    gfade = (cur == 0 ? 1 : smooth(0, 2.5f, st)) * smooth(d, d - 3, st);
    flashA = 0;
    lensOn = 0;
    scenes[cur].draw(st);
    flashA *= gfade;
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int windowed = 0, rootmode = 0, showfps = 0, shot = 0, start = 0;
    float speed = 1, shotT = 0; const char *shotFile = NULL; Window target = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;                   /* accept -opt and --opt */
        int more = i + 1 < argc, matched = 0;
        if (!strcmp(a, "-window")) windowed = 1;
        else if (!strcmp(a, "-root")) rootmode = 1;
        else if (!strcmp(a, "-window-id") && more) target = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "-fps")) showfps = 1;
        else if (!strcmp(a, "-scene") && more) start = atoi(argv[++i]);     /* -1 = random */
        else if (!strcmp(a, "-speed") && more) speed = clampf(atof(argv[++i]), 0.05f, 10);
        else if (!strcmp(a, "-shuffle")) shuffle = 1;
        else if (!strcmp(a, "-shot") && i + 3 < argc) { shot = 1; start = atoi(argv[++i]); shotT = atof(argv[++i]); shotFile = argv[++i]; windowed = 1; }
        else if (!strncmp(a, "-no-", 4)) {
            for (int s = 0; s < NSCENES; s++) if (!strcmp(a + 4, sceneName[s])) { enabled[s] = 0; matched = 1; }
            if (!matched) fprintf(stderr, "cosmogenesis: unknown scene in %s\n", argv[i]);
        }
        else {
            if (strcmp(a, "-h") && strcmp(a, "-help")) fprintf(stderr, "cosmogenesis: unknown option %s\n", argv[i]);
            printf("usage: %s [options]\n"
                   "  -window             run in a 1280x720 window (space/n next scene, p previous, q quit)\n"
                   "  -root               draw on the root window / XSCREENSAVER_WINDOW\n"
                   "  -window-id ID       draw into an existing window\n"
                   "  -scene N            start at scene N (0-%d), or -1 for a random one\n"
                   "  -speed X            time multiplier (default 1)\n"
                   "  -shuffle            play scenes in random order\n"
                   "  -no-NAME            leave a scene out; NAME is one of:\n"
                   "                      bigbang galaxy collision blackhole nursery planets supernova\n"
                   "  -fps                print the frame rate to stderr\n"
                   "  -shot N T file.ppm  render scene N at time T to an image and exit\n", argv[0], NSCENES - 1);
            return strcmp(a, "-h") && strcmp(a, "-help") ? 1 : 0;
        }
    }
    int anyOn = 0;
    for (int s = 0; s < NSCENES; s++) anyOn |= enabled[s];
    if (!anyOn) for (int s = 0; s < NSCENES; s++) enabled[s] = 1;   /* everything unticked: show it all */
    const char *env = getenv("XSCREENSAVER_WINDOW");
    if (!target && env && *env) target = strtoul(env, NULL, 0);

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "cosmogenesis: cannot open display\n"); return 1; }
    int scr = XDefaultScreen(dpy);
    Window root = XRootWindow(dpy, scr), win;
    if (!target && rootmode) target = root;
    XVisualInfo *vi;
    int ownwin = !target;

    if (target) {
        XWindowAttributes wa; XGetWindowAttributes(dpy, target, &wa);
        XVisualInfo tmpl; int n; tmpl.visualid = XVisualIDFromVisual(wa.visual);
        vi = XGetVisualInfo(dpy, 1 /* VisualIDMask */, &tmpl, &n);
        win = target;
    } else {
        int attr[] = { 4 /*RGBA*/, 5 /*DOUBLEBUFFER*/, 8, 8, 9, 8, 10, 8, 0 };
        vi = glXChooseVisual(dpy, scr, attr);
        if (!vi) { fprintf(stderr, "cosmogenesis: no suitable GLX visual\n"); return 1; }
        XSetWindowAttributes swa; memset(&swa, 0, sizeof swa);
        swa.colormap = XCreateColormap(dpy, root, vi->visual, 0);
        swa.event_mask = KeyPressMask | ButtonPressMask | PointerMotionMask | StructureNotifyMask;
        int w = windowed ? 1280 : XDisplayWidth(dpy, scr), h = windowed ? 720 : XDisplayHeight(dpy, scr);
        win = XCreateWindow(dpy, root, 0, 0, w, h, 0, vi->depth, 1, vi->visual,
                            CWColormap | CWEventMask | CWBorderPixel | CWBackPixel, &swa);
        XStoreName(dpy, win, "Cosmogenesis");
        Atom del = XInternAtom(dpy, "WM_DELETE_WINDOW", 0); XSetWMProtocols(dpy, win, &del, 1);
        if (!windowed) {
            Atom st_ = XInternAtom(dpy, "_NET_WM_STATE", 0), fs = XInternAtom(dpy, "_NET_WM_STATE_FULLSCREEN", 0);
            XChangeProperty(dpy, win, st_, 4 /*XA_ATOM*/, 32, 0, (unsigned char *)&fs, 1);
            static const char blank[1] = { 0 }; XColor black; memset(&black, 0, sizeof black);
            Pixmap bm = XCreateBitmapFromData(dpy, win, blank, 1, 1);
            XDefineCursor(dpy, win, XCreatePixmapCursor(dpy, bm, bm, &black, &black, 0, 0));
        }
        if (!shot) XMapRaised(dpy, win);
    }
    GLXContext ctx = glXCreateContext(dpy, vi, NULL, 1);
    if (!ctx || !glXMakeCurrent(dpy, win, ctx)) { fprintf(stderr, "cosmogenesis: cannot create GL context\n"); return 1; }
    int (*swapInterval)(int) = (int (*)(int))glXGetProcAddressARB((const unsigned char *)"glXSwapIntervalMESA");
    if (swapInterval) swapInterval(1);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH); glLineWidth(1.2f);
    gl_setup();
    rng ^= (unsigned)time(NULL) * 2654435761u; if (!rng) rng = 1;
    if (shot) rng = 0x2545F491u;
    init_stars();
    if (shot) start_scene(start);
    else {
        if (start < 0 || start >= NSCENES) {             /* random enabled scene */
            int n = 0, pick; for (int i = 0; i < NSCENES; i++) n += enabled[i];
            pick = (int)(frand() * n) % n;
            for (start = 0; !enabled[start] || pick-- > 0; start++) ;
        }
        while (!enabled[start]) start = (start + 1) % NSCENES;
        start_scene(start);
        if (shuffle) {                                   /* deck for the rest of this round */
            deal();
            for (int i = 0; i < deckN; i++) if (deck[i] == cur) { deck[i] = deck[0]; deck[0] = cur; }
            deckPos = 1;
        }
    }

    if (shot) {
        scrW = 1280; scrH = 720;
        float dt = 1 / 30.0f;
        while (st + dt < shotT) advance(dt);
        st = shotT;
        draw_scene(); render_frame(); glFinish();
        unsigned char *px = malloc(scrW * scrH * 3);
        glReadBuffer(GL_BACK); glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, scrW, scrH, GL_RGB, GL_UNSIGNED_BYTE, px);
        FILE *f = fopen(shotFile, "wb");
        if (!f) { perror(shotFile); return 1; }
        fprintf(f, "P6\n%d %d\n255\n", scrW, scrH);
        for (int y = scrH - 1; y >= 0; y--) fwrite(px + y * scrW * 3, 1, scrW * 3, f);
        fclose(f);
        return 0;
    }

    {   /* window size once; resize events keep it current (no per-frame X round trip) */
        XWindowAttributes wa; XGetWindowAttributes(dpy, win, &wa);
        scrW = wa.width > 0 ? wa.width : 1; scrH = wa.height > 0 ? wa.height : 1;
        if (!ownwin) XSelectInput(dpy, win, StructureNotifyMask);
    }
    double last = now(), fpsT = last; int frames = 0, mx0 = -1, my0 = -1;
    double startT = last;
    for (;;) {
        while (XPending(dpy)) {
            XEvent ev; XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) { scrW = ev.xconfigure.width > 0 ? ev.xconfigure.width : 1; scrH = ev.xconfigure.height > 0 ? ev.xconfigure.height : 1; continue; }
            if (!ownwin) continue;
            if (ev.type == ClientMessage) return 0;
            if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (windowed && (k == ' ' || k == 'n' || k == 0xff53 /*Right*/)) { step_scene(1); continue; }
                if (windowed && (k == 'p' || k == 0xff51 /*Left*/)) { step_scene(-1); continue; }
                if (!windowed || k == 'q' || k == 0xff1b) return 0;
            }
            if (!windowed && ev.type == ButtonPress) return 0;
            if (!windowed && ev.type == MotionNotify && now() - startT > 1.0) {
                if (mx0 < 0) { mx0 = ev.xkey.x_root; my0 = ev.xkey.y_root; }
                else if (abs(ev.xkey.x_root - mx0) + abs(ev.xkey.y_root - my0) > 20) return 0;
            }
        }
        double tnow = now(); float dt = (float)(tnow - last); last = tnow;
        if (dt > 0.1f) dt = 0.1f;
        advance(dt * speed);
        draw_scene(); render_frame();
        glXSwapBuffers(dpy, win);

        frames++;
        if (showfps && tnow - fpsT > 2) { fprintf(stderr, "scene %d t=%.1f  %.1f fps\n", cur, st, frames / (tnow - fpsT)); frames = 0; fpsT = tnow; }
        double spent = now() - tnow;            /* cap at ~60 fps if vsync is off */
        if (spent < 1 / 62.0) usleep((useconds_t)((1 / 62.0 - spent) * 1e6));
    }
}
