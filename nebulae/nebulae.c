/*
 * nebulae - slow fly-throughs of volumetric nebulae and stellar nurseries.
 *
 * Each nebula is a 3D volume (gas, dust, ionisation) generated on the GPU
 * from warped noise and shaped into one of three forms:
 *   - a cluster-carved cavity with dust pillars pointing at the stars
 *   - a wind-blown bubble around a hot star
 *   - a dark dust cloud with blue reflection nebulosity
 * Starlight is propagated through the volume once (with dust shadowing),
 * then every frame the volume is ray-marched with animated fine detail.
 * The camera tours the interesting spots; as it arrives, protostars ignite
 * inside dense cores, light up the gas around them and launch jets.
 *
 * Build:  gcc -O2 -o nebulae nebulae.c -lGL -lX11 -lm
 *
 * Usage:
 *   nebulae                  full screen, exits on key/mouse
 *   nebulae -window          1280x720 window (n = next nebula, q/Esc = quit)
 *   nebulae -root            for xscreensaver (draws on XSCREENSAVER_WINDOW)
 *   nebulae -quality Q       auto (default), low, medium, high, ultra
 *   nebulae -palette P       mixed (default), natural, hubble
 *   nebulae -duration S      seconds per nebula (default 130)
 *   nebulae -speed X         time multiplier
 *   nebulae -fps             print frame rate and quality level
 *   nebulae -shot SEED T f.ppm   render nebula SEED at time T and exit
 */
#define GL_GLEXT_PROTOTYPES 1
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
#define PI 3.14159265f
static unsigned int rng = 0x9E3779B9u;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return (rng >> 8) * (1.0f / 16777216.0f); }
static float srand1(void) { return frand() * 2 - 1; }
static float clampf(float x, float a, float b) { return x < a ? a : x > b ? b : x; }
static float smooth(float e0, float e1, float x) { float t = clampf((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
static float lerp(float a, float b, float t) { return a + (b - a) * t; }
static float hash1(int n) { unsigned int h = (unsigned)n * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; return (h & 0xffffff) / 16777216.0f; }
static void rand_unit(float *d) { float z = srand1(), a = frand() * 2 * PI, r = sqrtf(1 - z * z); d[0] = r * cosf(a); d[1] = z; d[2] = r * sinf(a); }
static float len3(const float *v) { return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
static void norm3(float *v) { float l = len3(v); if (l > 0) { v[0] /= l; v[1] /= l; v[2] /= l; } }
static void cross3(const float *a, const float *b, float *o) { o[0] = a[1]*b[2]-a[2]*b[1]; o[1] = a[2]*b[0]-a[0]*b[2]; o[2] = a[0]*b[1]-a[1]*b[0]; }
static float dot3(const float *a, const float *b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/* optical constants shared by the CPU and the shaders */
#define KD 9.0f      /* dust extinction per unit length at density 1 */
#define KG 0.6f      /* gas extinction */

/* ------------------------------------------------------------------ */
/* Shaders                                                             */
/* ------------------------------------------------------------------ */
static const char *fsVS =
"#version 330 core\n"
"out vec2 uv;\n"
"void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";

static const char *noiseGLSL =
"float hash(vec3 p) { p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419)); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }\n"
"float vnoise(vec3 x) {\n"
"  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);\n"
"  return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),\n"
"             mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);\n"
"}\n"
"float fbm(vec3 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec3(1.7, 9.2, 3.1); a *= 0.5; } return s / 0.96875; }\n"
"float ridged(vec3 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { float n = 1.0 - abs(vnoise(p) * 2.0 - 1.0); s += a * n * n; p = p * 2.07 + vec3(5.3, 1.1, 7.7); a *= 0.5; } return s / 0.96875; }\n";

/* density volume: r = gas, g = dust/2, b = ionisation (OIII fraction) */
static const char *genFS_body =
"in vec2 uv; out vec4 o;\n"
"uniform float zSlice; uniform float dim; uniform int kind;\n"
"uniform vec3 seed, cluster; uniform float cavR;\n"
"uniform vec4 pil[6]; uniform int nPil;\n"
"void main() {\n"
"  vec3 p = vec3(gl_FragCoord.xy / dim, zSlice) * 2.0 - 1.0;\n"
"  vec3 w = vec3(fbm(p * 1.6 + seed), fbm(p * 1.6 + seed + 17.3), fbm(p * 1.6 + seed + 41.7)) - 0.5;\n"
"  vec3 q = p + w * 0.6;\n"
"  float n = fbm(q * 2.9 + seed * 1.3);\n"
"  float r = ridged(q * 3.1 + seed * 0.7);\n"
"  float env = smoothstep(1.0, 0.5, length(p * vec3(0.95, 1.2, 0.95)));\n"
"  float gas, dust, oxy;\n"
"  float dc = length(p - cluster);\n"
"  if (kind == 0) {\n"
"    float edge = cavR * (1.0 + 0.4 * (n - 0.5));\n"
"    gas = clamp(n * 3.0 - 1.3, 0.0, 1.0) * env * smoothstep(edge * 0.85, edge * 1.15, dc);\n"
"    float rim = exp(-pow((dc - edge) / 0.075, 2.0)) * env * (0.35 + 0.9 * n);\n"
"    gas += rim * 0.9;\n"
"    dust = gas * (0.25 + 1.2 * r) * smoothstep(edge * 0.95, edge * 1.5, dc);\n"
"    for (int i = 0; i < nPil; i++) {\n"
"      vec3 dir = pil[i].xyz, b = cluster + dir * edge * 1.75, tip = cluster + dir * cavR * pil[i].w;\n"
"      vec3 pa = p - tip, ba = b - tip; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);\n"
"      float d = length(pa - ba * h) + 0.035 * (r - 0.5) + 0.02 * (n - 0.5);\n"
"      float rad = mix(0.03, 0.1, h);\n"
"      float pl = smoothstep(rad, rad * 0.35, d) * env;\n"
"      gas = max(gas, pl * 0.85); dust = max(dust, pl * (1.1 + 0.6 * r));\n"
"    }\n"
"    gas += 0.05 * n * smoothstep(edge, 0.0, dc) * env;\n"
"    oxy = smoothstep(edge * 1.6, edge * 0.45, dc) * (0.45 + 0.55 * n);\n"
"  } else if (kind == 1) {\n"
"    float R = cavR * (1.0 + 0.3 * (n - 0.5));\n"
"    float shell = exp(-pow((dc - R) / (0.04 + 0.06 * r), 2.0)) * (0.5 + 0.8 * n);\n"
"    gas = shell + clamp(n * 3.0 - 1.35, 0.0, 1.0) * env * smoothstep(R * 0.9, R * 1.35, dc);\n"
"    gas += 0.06 * smoothstep(R, 0.0, dc) * n;\n"
"    dust = clamp(r * 1.5 - 0.72, 0.0, 1.0) * env * smoothstep(R * 1.05, R * 1.6, dc) * 1.5 + shell * 0.15;\n"
"    oxy = smoothstep(R * 1.2, R * 0.55, dc);\n"
"  } else {\n"
"    dust = clamp(r * 2.2 - 1.0, 0.0, 1.0) * env * 1.7 + clamp(n * 2.2 - 1.0, 0.0, 1.0) * env * 0.6;\n"
"    gas = 0.35 * clamp(n * 2.4 - 1.0, 0.0, 1.0) * env;\n"
"    oxy = 0.25 * n;\n"
"  }\n"
"  oxy = max(oxy, 0.58 * smoothstep(0.56, 0.78, fbm(q * 1.4 + seed + 77.0)));   /* patches of doubly ionised oxygen */\n"
"  oxy *= 1.0 - smoothstep(0.25, 0.8, dust);   /* dense, shielded gas stays neutral: no OIII */\n"
"  float lanes = ridged(q * 2.2 + seed * 2.1 + 31.0);\n"
"  dust += clamp(lanes * 2.6 - 1.7, 0.0, 1.0) * env * (kind == 2 ? 0.6 : 1.2) * smoothstep(cavR * 0.8, cavR * 1.3, dc + (kind == 2 ? 9.0 : 0.0));\n"
"  o = vec4(clamp(gas, 0.0, 1.0), clamp(dust * 0.5, 0.0, 1.0), clamp(oxy, 0.0, 1.0), 1.0);\n"
"}\n";

/* starlight volume: colour of the light reaching each cell, shadowed by dust */
static const char *lightFS =
"#version 330 core\n"
"in vec2 uv; out vec4 o;\n"
"uniform sampler3D vol; uniform float zSlice; uniform float dim;\n"
"uniform vec3 lp[8]; uniform vec3 lc[8]; uniform int nL; uniform float kD, kG;\n"
"void main() {\n"
"  vec3 p = vec3(gl_FragCoord.xy / dim, zSlice) * 2.0 - 1.0;\n"
"  vec3 L = vec3(0.0);\n"
"  for (int k = 0; k < nL; k++) {\n"
"    vec3 d = lp[k] - p; float dist = length(d); vec3 dir = d / max(dist, 1e-4);\n"
"    float ds = dist / 40.0, tau = 0.0;\n"
"    for (int j = 0; j < 40; j++) {\n"
"      vec3 q = p + dir * ds * (float(j) + 0.5);\n"
"      vec4 v = texture(vol, q * 0.5 + 0.5);\n"
"      tau += (v.g * 2.0 * kD + v.r * kG) * ds;\n"
"    }\n"
"    L += lc[k] / (dist * dist + 0.025) * exp(-tau);\n"
"  }\n"
"  o = vec4(L, 1.0);\n"
"}\n";

static const char *marchFS =
"#version 330 core\n"
"in vec2 uv; out vec4 o;\n"
"uniform sampler3D vol, detail;      /* vol: gas, dust/2, ionisation, sqrt(starlight/12) */\n"
"uniform vec3 camPos, camR, camU, camF; uniform float tanX, tanY;\n"
"uniform int steps; uniform float time, pixAng;\n"
"uniform vec3 colHa, colO3, colRefl, lightTint;\n"
"uniform float kD, kG, emGain, reflGain, ambient;\n"
"uniform vec4 dl[5];\n"
"uniform int rowParity;   /* -1: all rows; 0/1: only even/odd rows this frame */\n"
"float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }\n"
"void main() {\n"
"  if (rowParity >= 0 && (int(gl_FragCoord.y) & 1) != rowParity) discard;\n"
"  vec3 rd = normalize(camF + (uv.x * 2.0 - 1.0) * tanX * camR + (uv.y * 2.0 - 1.0) * tanY * camU);\n"
"  vec3 inv = 1.0 / rd, t0 = (-1.0 - camPos) * inv, t1 = (1.0 - camPos) * inv;\n"
"  vec3 tmin = min(t0, t1), tmax = max(t0, t1);\n"
"  float tn = max(max(tmin.x, tmin.y), tmin.z), tf = min(min(tmax.x, tmax.y), tmax.z);\n"
"  if (tf <= max(tn, 0.0)) { o = vec4(0.0, 0.0, 0.0, 1.0); return; }\n"
"  tn = max(tn, 0.0);\n"
"  float dt = 3.4641 / float(steps);\n"
"  float t = tn + dt * ign(gl_FragCoord.xy);\n"
"  vec3 acc = vec3(0.0); float T = 1.0;\n"
"  vec3 flow = vec3(time * 0.004, time * 0.0025, -time * 0.003);\n"
"  for (int i = 0; i < steps; i++) {\n"
"    if (t >= tf) break;\n"
"    vec3 p = camPos + rd * t;\n"
"    vec4 v = texture(vol, p * 0.5 + 0.5);\n"
"    if (v.r + v.g < 0.004) { t += dt * 2.5; continue; }      /* skip through empty space */\n"
"    float fp = t * pixAng + dt * 0.25;   /* footprint of this sample, for mip selection */\n"
"    vec2 d1 = textureLod(detail, p * 2.1 + flow, log2(max(fp * 2.1 * 64.0, 1.0))).rg;\n"
"    vec2 d2 = textureLod(detail, p * 6.3 - flow * 1.7, log2(max(fp * 6.3 * 64.0, 1.0))).rg;\n"
"    float m = clamp(1.3 * d1.r + 0.7 * d2.r - 0.45, 0.0, 2.0);\n"
"    float fil = 0.35 + 1.0 * d2.g + 0.5 * d1.g;\n"
"    float nearF = smoothstep(0.0, 0.05, t);\n"
"    float gas = smoothstep(0.12, 0.65, v.r * m * fil) * 0.9 * nearF;\n"
"    float dust = v.g * 2.0 * smoothstep(0.15, 0.95, m * (0.4 + 0.5 * fil)) * 1.5 * nearF;\n"
"    vec3 L = lightTint * (v.a * v.a * 12.0);\n"
"    float Ld = 0.0;   /* newborn stars excite hydrogen in wisps along the filaments */\n"
"    for (int k = 0; k < 5; k++) { vec3 dd = dl[k].xyz - p; Ld += dl[k].w / (dot(dd, dd) + 0.002); }\n"
"    vec3 ec = mix(colHa, colO3, smoothstep(0.25, 0.75, v.b));\n"
"    vec3 em = gas * ec * (ambient + emGain * L) + dust * reflGain * L * colRefl;\n"
"    em += (gas + 0.3 * dust) * fil * fil * Ld * emGain * (colHa + vec3(0.15, 0.1, 0.05));\n"
"    float sig = dust * kD + gas * kG;\n"
"    acc += T * em * dt;\n"
"    T *= exp(-sig * dt);\n"
"    if (T < 0.01) break;\n"
"    t += dt;\n"
"  }\n"
"  o = vec4(acc, T);\n"
"}\n";

/* writes starlight luminance into the density volume's alpha channel */
static const char *bakeFS =
"#version 330 core\n"
"in vec2 uv; out vec4 o; uniform sampler3D lightVol; uniform float zSlice; uniform float dim; uniform vec3 tint;\n"
"void main() { vec3 L = texture(lightVol, vec3(gl_FragCoord.xy / dim, zSlice)).rgb; o = vec4(0.0, 0.0, 0.0, sqrt(clamp(dot(L, vec3(0.333)) / 12.0, 0.0, 1.0))); }\n";

static const char *spriteVS =
"#version 330 core\n"
"layout(location=0) in vec2 corner;\n"
"layout(location=1) in vec4 posSize;\n"
"layout(location=2) in vec4 color;\n"
"layout(location=3) in float spike;\n"
"uniform mat4 V, P;\n"
"out vec2 uv; out vec4 col; out float spk;\n"
"void main() {\n"
"  vec4 vp = V * vec4(posSize.xyz, 1.0);\n"
"  gl_Position = P * (vp + vec4(corner * posSize.w * 0.5, 0.0, 0.0));\n"
"  uv = corner; col = color; spk = spike;\n"
"}\n";

static const char *spriteFS =
"#version 330 core\n"
"in vec2 uv; in vec4 col; in float spk; out vec4 o;\n"
"uniform sampler2D nebT; uniform float nebK; uniform vec2 scr;   /* nebK > 0: behind the nebula */\n"
"void main() {\n"
"  float r2 = dot(uv, uv); if (r2 > 1.0) discard;\n"
"  float a = exp(-r2 * 30.0) + 0.12 * exp(-r2 * 6.0) * (1.0 - r2);\n"
"  a += spk * 0.5 * (exp(-abs(uv.y) * 90.0) * (1.0 - abs(uv.x)) + exp(-abs(uv.x) * 90.0) * (1.0 - abs(uv.y)));\n"
"  if (nebK > 0.0) a *= mix(1.0, texture(nebT, gl_FragCoord.xy / scr).a, nebK);\n"
"  o = vec4(col.rgb * col.a * a, 1.0);\n"
"}\n";

static const char *brightFS =
"#version 330 core\n"
"in vec2 uv; out vec4 o; uniform sampler2D src, neb; uniform vec2 texel; uniform float nebK;\n"
"void main() {\n"
"  vec3 c = texture(src, uv + texel * vec2(-1, -1)).rgb + texture(src, uv + texel * vec2(1, -1)).rgb\n"
"         + texture(src, uv + texel * vec2(-1, 1)).rgb + texture(src, uv + texel * vec2(1, 1)).rgb;\n"
"  c = c * 0.25 + texture(neb, uv).rgb * nebK; float l = max(c.r, max(c.g, c.b));\n"
"  o = vec4(c * (max(l - 0.6, 0.0) / max(l, 1e-4)), 1.0);\n"
"}\n";

static const char *blurFS =
"#version 330 core\n"
"in vec2 uv; out vec4 o; uniform sampler2D src; uniform vec2 dir;\n"
"void main() {\n"
"  vec3 c = texture(src, uv).rgb * 0.227;\n"
"  c += (texture(src, uv + dir * 1.385).rgb + texture(src, uv - dir * 1.385).rgb) * 0.316;\n"
"  c += (texture(src, uv + dir * 3.231).rgb + texture(src, uv - dir * 3.231).rgb) * 0.070;\n"
"  o = vec4(c, 1.0);\n"
"}\n";

static const char *postFS =
"#version 330 core\n"
"in vec2 uv; out vec4 o;\n"
"uniform sampler2D hdr, bloom, neb; uniform float exposure, fade, nebK; uniform vec2 nebTexel;\n"
"vec3 aces(vec3 x) { return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }\n"
"void main() {\n"
"  vec3 n = 0.25 * (texture(neb, uv + nebTexel * vec2(-0.6, -0.6)).rgb + texture(neb, uv + nebTexel * vec2(0.6, -0.6)).rgb\n"
"                 + texture(neb, uv + nebTexel * vec2(-0.6, 0.6)).rgb + texture(neb, uv + nebTexel * vec2(0.6, 0.6)).rgb);\n"
"  vec3 c = texture(hdr, uv).rgb + n * nebK + texture(bloom, uv).rgb * 0.45;\n"
"  float l = dot(c, vec3(0.2126, 0.7152, 0.0722)); c = max(mix(vec3(l), c, 1.3), 0.0);   /* restore saturation ACES takes away */\n"
"  c = aces(max(c * exposure - 0.004, 0.0)) * fade;\n"
"  c = pow(c, vec3(1.0 / 1.7));\n"
"  c += (fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) - 0.5) / 255.0;\n"
"  o = vec4(c, 1.0);\n"
"}\n";

static GLuint compile(GLenum type, const char *a, const char *b, const char *c)
{
    const char *src[3] = { a, b ? b : "", c ? c : "" };
    GLuint s = glCreateShader(type); GLint ok;
    glShaderSource(s, 3, src, NULL); glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof log, NULL, log); fprintf(stderr, "nebulae: shader error:\n%s\n", log); exit(1); }
    return s;
}

static GLuint program(const char *vs, const char *fs0, const char *fs1, const char *fs2)
{
    GLuint p = glCreateProgram(); GLint ok;
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs, NULL, NULL));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs0, fs1, fs2));
    glLinkProgram(p); glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof log, NULL, log); fprintf(stderr, "nebulae: link error:\n%s\n", log); exit(1); }
    return p;
}
#define U(p, n) glGetUniformLocation(p, n)

/* ------------------------------------------------------------------ */
/* GL resources                                                        */
/* ------------------------------------------------------------------ */
#define VD 192          /* density volume resolution */
#define LD 64           /* light volume resolution */
#define DD 64           /* tiling detail noise */

static GLuint pBake, pGen, pLight, pMarch, pSprite, pBright, pBlur, pPost;
static GLuint volTex, lightTex, detailTex, vaoEmpty, vaoSprite, quadVbo;
static GLuint fboHdr, texHdr, fboNeb, texNeb, fboB[2], texB[2], fboTmp;
static int scrW = 1280, scrH = 720, fbW = 0, fbH = 0, nebW = 0, nebH = 0;
static GLuint tq[2]; static int tqUsed[2] = { 0, 0 }, tqFrame = 0;
static float marchMs = 0;     /* smoothed GPU time of a whole frame */
static int nebFresh = 1;   /* march target must be redrawn before reuse */
static GLuint pboVol, pboLight;

static unsigned char *volCPU;
static float *lightCPU;

static void make_rt(GLuint *fb, GLuint *tex, int w, int h, GLint fmt)
{
    if (*fb) { glDeleteFramebuffers(1, fb); glDeleteTextures(1, tex); }
    glGenTextures(1, tex); glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, fb); glBindFramebuffer(GL_FRAMEBUFFER, *fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "nebulae: float render targets unsupported\n"); exit(1); }
}

static GLuint make_tex3d(int n, GLint ifmt, GLenum fmt, GLenum type, const void *data, GLint wrap)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_3D, t);
    glTexImage3D(GL_TEXTURE_3D, 0, ifmt, n, n, n, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap);
    return t;
}

/* periodic smooth noise for the animated fine structure */
static void make_detail(void)
{
    int n = DD * DD * DD;
    float *a = malloc(sizeof(float) * n), *b = malloc(sizeof(float) * n), *acc = calloc(n, sizeof(float));
    for (int oct = 0; oct < 2; oct++) {
        int R = oct ? 1 : 3;
        for (int i = 0; i < n; i++) a[i] = frand();
        for (int pass = 0; pass < 2; pass++)
            for (int ax = 0; ax < 3; ax++) {
                int st = ax == 0 ? 1 : ax == 1 ? DD : DD * DD;
                for (int i = 0; i < n; i++) {
                    int c = (i / st) % DD, base = i - c * st; float s = 0;
                    for (int k = -R; k <= R; k++) s += a[base + ((c + k + DD) % DD) * st];
                    b[i] = s / (2 * R + 1);
                }
                memcpy(a, b, sizeof(float) * n);
            }
        float mn = 1e9, mx = -1e9;
        for (int i = 0; i < n; i++) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; }
        for (int i = 0; i < n; i++) { a[i] = (a[i] - mn) / (mx - mn); acc[i] += a[i] * (oct ? 0.35f : 0.65f); }
    }
    unsigned char *u = malloc(n * 2);
    for (int i = 0; i < n; i++) {
        float v = clampf(acc[i], 0, 1), r = 1 - fabsf(2 * (a[i] - 0.5f)) * 2;   /* ridges of the finest octave */
        r = clampf(r, 0, 1); r = r * r * r;
        u[2 * i] = (unsigned char)(v * 255); u[2 * i + 1] = (unsigned char)(r * 255);
    }
    detailTex = make_tex3d(DD, GL_RG8, GL_RG, GL_UNSIGNED_BYTE, u, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_3D);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    free(a); free(b); free(acc); free(u);
}

static void gl_setup(void)
{
    pGen    = program(fsVS, "#version 330 core\n", noiseGLSL, genFS_body);
    pLight  = program(fsVS, lightFS, NULL, NULL);
    pMarch  = program(fsVS, marchFS, NULL, NULL);
    pBake   = program(fsVS, bakeFS, NULL, NULL);
    pSprite = program(spriteVS, spriteFS, NULL, NULL);
    pBright = program(fsVS, brightFS, NULL, NULL);
    pBlur   = program(fsVS, blurFS, NULL, NULL);
    pPost   = program(fsVS, postFS, NULL, NULL);

    glGenVertexArrays(1, &vaoEmpty);
    static const float quad[8] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    glGenVertexArrays(1, &vaoSprite); glBindVertexArray(vaoSprite);
    glGenBuffers(1, &quadVbo); glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    for (int i = 1; i <= 3; i++) { glEnableVertexAttribArray(i); glVertexAttribDivisor(i, 1); }
    glBindVertexArray(0);

    volTex = make_tex3d(VD, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, NULL, GL_CLAMP_TO_EDGE);
    lightTex = make_tex3d(LD, GL_RGBA16F, GL_RGBA, GL_FLOAT, NULL, GL_CLAMP_TO_EDGE);
    make_detail();
    glGenFramebuffers(1, &fboTmp);
    glGenQueries(2, tq);
    glGenBuffers(1, &pboVol); glBindBuffer(GL_PIXEL_PACK_BUFFER, pboVol);
    glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)VD * VD * VD * 4, NULL, GL_STREAM_READ);
    glGenBuffers(1, &pboLight); glBindBuffer(GL_PIXEL_PACK_BUFFER, pboLight);
    glBufferData(GL_PIXEL_PACK_BUFFER, sizeof(float) * LD * LD * LD * 4, NULL, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    volCPU = malloc((size_t)VD * VD * VD * 4);
    lightCPU = malloc(sizeof(float) * LD * LD * LD * 4);
    glDisable(GL_DEPTH_TEST);
}

/* ------------------------------------------------------------------ */
/* Quality levels (ray-march resolution and step count)                */
/* ------------------------------------------------------------------ */
/* resolution scale, steps, and whether rows are split across two frames (1) or all marched each frame */
static const float qScale[] = { 0.25f, 0.33f, 0.4f, 0.5f, 0.6f, 0.75f, 1.0f };
static const int qSteps[]   = { 44, 52, 64, 80, 96, 112, 128 };
static const int qEvery[]   = { 2, 2, 2, 1, 1, 1, 1 };
#define NQ 7
static int qLevel = 3, qAuto = 1;
static double qLastChange = 0;

static void ensure_targets(void)
{
    int nw = (int)(scrW * qScale[qLevel] + 0.5f), nh = (int)(scrH * qScale[qLevel] + 0.5f);
    if (nw < 16) nw = 16;
    if (nh < 16) nh = 16;
    if (fbW != scrW || fbH != scrH) {
        /* screen-size buffers need no alpha: 32-bit packed float halves their bandwidth */
        make_rt(&fboHdr, &texHdr, scrW, scrH, GL_R11F_G11F_B10F);
        make_rt(&fboB[0], &texB[0], (scrW + 3) / 4, (scrH + 3) / 4, GL_R11F_G11F_B10F);
        make_rt(&fboB[1], &texB[1], (scrW + 3) / 4, (scrH + 3) / 4, GL_R11F_G11F_B10F);
        fbW = scrW; fbH = scrH;
    }
    if (nebW != nw || nebH != nh) { make_rt(&fboNeb, &texNeb, nw, nh, GL_RGBA16F);   /* alpha holds transmittance */ nebW = nw; nebH = nh; nebFresh = 1; }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

/* ------------------------------------------------------------------ */
/* Nebula description and generation                                   */
/* ------------------------------------------------------------------ */
typedef struct {
    int kind, palette;
    float seed[3], cluster[3], cavR;
    int nPil; float pil[6][4];
    int nL; float lp[8][3], lc[8][3];
    float colHa[3], colO3[3], colRefl[3], emGain, reflGain, ambient, exposure;
} Nebula;
static Nebula N;
static int paletteOpt = -1;     /* -1 mixed, 0 natural, 1 hubble */

static void vsample(const float *p, float *g, float *d)
{
    if (fabsf(p[0]) >= 1 || fabsf(p[1]) >= 1 || fabsf(p[2]) >= 1) { *g = *d = 0; return; }
    int ix = (int)((p[0] * 0.5f + 0.5f) * VD), iy = (int)((p[1] * 0.5f + 0.5f) * VD), iz = (int)((p[2] * 0.5f + 0.5f) * VD);
    const unsigned char *v = volCPU + 4 * ((size_t)ix + VD * ((size_t)iy + (size_t)VD * iz));
    *g = v[0] / 255.0f; *d = v[1] / 255.0f * 2;
}

static float lsample(const float *p)
{
    if (fabsf(p[0]) >= 1 || fabsf(p[1]) >= 1 || fabsf(p[2]) >= 1) return 0;
    int ix = (int)((p[0] * 0.5f + 0.5f) * LD), iy = (int)((p[1] * 0.5f + 0.5f) * LD), iz = (int)((p[2] * 0.5f + 0.5f) * LD);
    const float *v = lightCPU + 4 * (ix + LD * (iy + LD * iz));
    return v[0] + v[1] + v[2];
}

/* optical depth between two points (mean detail modulation folded in) */
static float tau_seg(const float *a, const float *b, int n)
{
    float d[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] }, L = len3(d), tau = 0, ds = L / n;
    for (int i = 0; i < n; i++) {
        float f = (i + 0.5f) / n, p[3] = { a[0] + d[0]*f, a[1] + d[1]*f, a[2] + d[2]*f }, g, du;
        vsample(p, &g, &du);
        tau += (du * KD * 0.85f + g * KG) * ds;
    }
    return tau;
}

static void set_palette(void)
{
    int hub = paletteOpt >= 0 ? paletteOpt : frand() < 0.35f;
    N.palette = hub;
    float tint = srand1() * 0.08f;
    if (!hub) {
        float ha[3] = { 1.0f, 0.13f + tint, 0.12f }, o3[3] = { 0.06f, 0.5f, 0.46f + tint }, rf[3] = { 0.3f, 0.5f, 1.0f };
        memcpy(N.colHa, ha, sizeof ha); memcpy(N.colO3, o3, sizeof o3); memcpy(N.colRefl, rf, sizeof rf);
    } else {
        float ha[3] = { 1.0f, 0.55f + tint, 0.06f }, o3[3] = { 0.02f, 0.42f, 1.0f }, rf[3] = { 0.4f, 0.6f, 1.0f };
        memcpy(N.colHa, ha, sizeof ha); memcpy(N.colO3, o3, sizeof o3); memcpy(N.colRefl, rf, sizeof rf);
    }
}

static void add_light(const float *p, float r, float g, float b, float I)
{
    if (N.nL >= 8) return;
    memcpy(N.lp[N.nL], p, sizeof(float) * 3);
    N.lc[N.nL][0] = r * I; N.lc[N.nL][1] = g * I; N.lc[N.nL][2] = b * I; N.nL++;
}

static void design_nebula(void)
{
    memset(&N, 0, sizeof N);
    float k = frand();
    N.kind = k < 0.45f ? 0 : k < 0.75f ? 1 : 2;
    for (int i = 0; i < 3; i++) N.seed[i] = frand() * 100;
    set_palette();
    rand_unit(N.cluster); float cr = 0.22f * frand();
    for (int i = 0; i < 3; i++) N.cluster[i] *= cr;
    N.emGain = 2.4f; N.reflGain = 0.03f; N.ambient = 0.012f; N.exposure = 1.4f;
    if (N.kind == 0) {
        N.cavR = 0.33f + 0.15f * frand();
        float hemi[3]; rand_unit(hemi);
        N.nPil = 3 + (int)(frand() * 3);
        for (int i = 0; i < N.nPil; i++) {
            float d[3]; rand_unit(d);
            for (int j = 0; j < 3; j++) d[j] = hemi[j] + 0.55f * d[j];
            norm3(d); memcpy(N.pil[i], d, sizeof d); N.pil[i][3] = 0.65f + 0.3f * frand();
        }
        for (int i = 0; i < 6; i++) {
            float p[3]; rand_unit(p); float r = 0.05f * cbrtf(frand());
            for (int j = 0; j < 3; j++) p[j] = N.cluster[j] + p[j] * r;
            add_light(p, 0.75f, 0.85f, 1.0f, 0.03f + 0.035f * frand());
        }
    } else if (N.kind == 1) {
        N.cavR = 0.4f + 0.15f * frand();
        add_light(N.cluster, 0.7f, 0.82f, 1.0f, 0.12f);
        float p[3]; rand_unit(p); for (int j = 0; j < 3; j++) p[j] = N.cluster[j] + p[j] * 0.04f;
        add_light(p, 1.0f, 0.95f, 0.9f, 0.03f);
    } else {
        N.reflGain = 3.0f; N.emGain = 1.6f; N.ambient = 0.06f; N.exposure = 2.0f;
        for (int i = 0; i < 3; i++) {
            float p[3]; rand_unit(p); float r = 0.25f + 0.4f * frand();
            for (int j = 0; j < 3; j++) p[j] *= r;
            add_light(p, 0.85f, 0.9f, 1.0f, 0.02f + 0.03f * frand());
        }
    }
}

static void fs_draw(void) { glBindVertexArray(vaoEmpty); glDrawArrays(GL_TRIANGLES, 0, 3); }

/* ------------------------------------------------------------------ */
/* Volume generation. Runs in chunks so a continuous show can build the */
/* next nebula while the current frame rate is kept up.                */
/* ------------------------------------------------------------------ */
static int genStage = -1, genZ = 0;   /* 0 density, 1 light, 2 bake, 3 readback, 4 map, 5 done */
static void gen_begin(void) { genStage = 0; genZ = 0; }

static int gen_step(int all)
{
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, fboTmp);
    do {
        if (genStage == 0) {                       /* density, 8 slices per call */
            glUseProgram(pGen);
            glUniform1f(U(pGen, "dim"), VD); glUniform1i(U(pGen, "kind"), N.kind);
            glUniform3fv(U(pGen, "seed"), 1, N.seed); glUniform3fv(U(pGen, "cluster"), 1, N.cluster);
            glUniform1f(U(pGen, "cavR"), N.cavR); glUniform1i(U(pGen, "nPil"), N.nPil);
            glUniform4fv(U(pGen, "pil"), 6, &N.pil[0][0]);
            glViewport(0, 0, VD, VD);
            for (int k = 0; k < 8 && genZ < VD; k++, genZ++) {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, volTex, 0, genZ);
                glUniform1f(U(pGen, "zSlice"), (genZ + 0.5f) / VD);
                fs_draw();
            }
            if (genZ >= VD) { genStage = 1; genZ = 0; }
        } else if (genStage == 1) {                /* starlight with dust shadows, 3 slices per call */
            glUseProgram(pLight);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, volTex); glUniform1i(U(pLight, "vol"), 0);
            glUniform1f(U(pLight, "dim"), LD); glUniform1i(U(pLight, "nL"), N.nL);
            glUniform3fv(U(pLight, "lp"), 8, &N.lp[0][0]); glUniform3fv(U(pLight, "lc"), 8, &N.lc[0][0]);
            glUniform1f(U(pLight, "kD"), KD); glUniform1f(U(pLight, "kG"), KG);
            glViewport(0, 0, LD, LD);
            for (int k = 0; k < 3 && genZ < LD; k++, genZ++) {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightTex, 0, genZ);
                glUniform1f(U(pLight, "zSlice"), (genZ + 0.5f) / LD);
                fs_draw();
            }
            if (genZ >= LD) { genStage = 2; genZ = 0; }
        } else if (genStage == 2) {                /* light luminance into the density alpha */
            glUseProgram(pBake);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, lightTex); glUniform1i(U(pBake, "lightVol"), 0);
            glUniform1f(U(pBake, "dim"), VD);
            glViewport(0, 0, VD, VD);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
            for (int k = 0; k < 32 && genZ < VD; k++, genZ++) {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, volTex, 0, genZ);
                glUniform1f(U(pBake, "zSlice"), (genZ + 0.5f) / VD);
                fs_draw();
            }
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            if (genZ >= VD) { genStage = 3; genZ = 0; }
        } else if (genStage == 3) {                /* CPU copies for planning and star extinction */
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            if (all) {
                glBindTexture(GL_TEXTURE_3D, volTex); glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_UNSIGNED_BYTE, volCPU);
                glBindTexture(GL_TEXTURE_3D, lightTex); glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, lightCPU);
                genStage = 5;
            } else {                               /* asynchronous: read into buffers, map next frame */
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboVol);
                glBindTexture(GL_TEXTURE_3D, volTex); glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pboLight);
                glBindTexture(GL_TEXTURE_3D, lightTex); glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, 0);
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                genStage = 4;
                break;
            }
        } else if (genStage == 4) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pboVol);
            void *m = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
            if (m) memcpy(volCPU, m, (size_t)VD * VD * VD * 4);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pboLight);
            m = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
            if (m) memcpy(lightCPU, m, sizeof(float) * LD * LD * LD * 4);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            genStage = 5;
        }
    } while (all && genStage != 5);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return genStage == 5;
}

/* ------------------------------------------------------------------ */
/* The tour: candidate sites, camera path, protostars                  */
/* ------------------------------------------------------------------ */
#define MAXW 48
static float wp[MAXW][3], wpT[MAXW][3], segStart[MAXW];
static int nwp;
static float tourDur = 130, tourOpt = 130;
static int durGiven = 0, continuous = 0;
static float tn = 0;                 /* show time (continuous mode: never resets) */

typedef struct { float p[3], s, lastUse; } Cand;
static Cand cand[8000]; static int nc;

#define MAXPS 32
typedef struct { float p[3], axis[3], tIgn; int id; } Proto;
static Proto ps[MAXPS]; static int nps = 0, psNext = 0, psIds = 0;

#define NFIELD 500
static float field[NFIELD][7];

static void add_proto(const float *p, float tIgn)
{
    Proto *P = &ps[psNext]; psNext = (psNext + 1) % MAXPS; if (nps < MAXPS) nps++;
    memcpy(P->p, p, 12); rand_unit(P->axis); P->tIgn = tIgn; P->id = psIds++;
}

static int near_proto(const float *p, float d)
{
    for (int i = 0; i < nps; i++) { float dx = p[0]-ps[i].p[0], dy = p[1]-ps[i].p[1], dz = p[2]-ps[i].p[2]; if (dx*dx + dy*dy + dz*dz < d*d) return 1; }
    return 0;
}

static int cmp_cand(const void *a, const void *b) { float d = ((const Cand *)b)->s - ((const Cand *)a)->s; return d > 0 ? 1 : d < 0 ? -1 : 0; }

/* dense dust that is also lit: bright-rimmed cores where stars are born */
static void find_cands(void)
{
    nc = 0;
    const int G = 20;
    for (int z = 0; z < G; z++) for (int y = 0; y < G; y++) for (int x = 0; x < G; x++) {
        float p[3] = { -0.8f + 1.6f * (x + frand()) / G, -0.8f + 1.6f * (y + frand()) / G, -0.8f + 1.6f * (z + frand()) / G }, g, d;
        vsample(p, &g, &d);
        if (d < 0.3f) continue;
        memcpy(cand[nc].p, p, 12);
        cand[nc].s = d * (0.3f + g) * clampf(lsample(p), 0.05f, 2.5f) * (0.7f + 0.6f * frand());
        cand[nc].lastUse = -1e9f;
        nc++;
    }
    qsort(cand, nc, sizeof(Cand), cmp_cand);
}

static void pick_view(const float *P, float dist, float *view)
{
    float best = -1e9;
    for (int tries = 0; tries < 48; tries++) {
        float d[3]; rand_unit(d);
        float c[3] = { P[0] + d[0]*dist, P[1] + d[1]*dist, P[2] + d[2]*dist }, g, du;
        if (len3(c) > 1.4f) continue;
        vsample(c, &g, &du);
        float tau = tau_seg(c, P, 32);
        /* prefer: clear air at the camera, a clear sight line, the lit side of the target */
        float tl[3] = { N.cluster[0]-P[0], N.cluster[1]-P[1], N.cluster[2]-P[2] }; norm3(tl);
        float sc = -3 * du - 1.2f * g - 0.6f * tau + 0.8f * dot3(d, tl) + 0.3f * srand1();
        if (sc > best) { best = sc; memcpy(view, c, 12); }
    }
    if (best == -1e9) for (int j = 0; j < 3; j++) view[j] = P[j] * 1.4f + 0.1f;
}

static void make_field(void)
{
    for (int i = 0; i < NFIELD; i++) {
        for (int j = 0; j < 3; j++) field[i][j] = srand1() * 1.1f;
        float t = frand();
        field[i][3] = t < 0.3f ? 1.0f : 0.85f; field[i][4] = t < 0.3f ? 0.8f : 0.9f; field[i][5] = t < 0.3f ? 0.6f : 1.0f;
        field[i][6] = 0.15f + 2.0f * powf(frand(), 6);
    }
}

static void entry_point(const float *towards, float r, float *o)
{
    float in[3]; memcpy(in, towards, 12); norm3(in);
    float jit[3]; rand_unit(jit);
    for (int j = 0; j < 3; j++) o[j] = (in[j] + 0.4f * jit[j]) * r;
}

/* the classic tour: approach, 3-5 stops, drift back out */
static void plan_tour(void)
{
    find_cands();
    nps = psNext = 0;
    float stops[6][3]; int core[6], ns = 0;
    int wantCores = N.kind == 2 ? 4 : 3;
    for (int i = 0; i < nc && ns < wantCores; i++) {
        int ok = 1;
        for (int k = 0; k < ns; k++) { float dx = cand[i].p[0]-stops[k][0], dy = cand[i].p[1]-stops[k][1], dz = cand[i].p[2]-stops[k][2]; if (dx*dx + dy*dy + dz*dz < 0.42f * 0.42f) ok = 0; }
        if (ok) { memcpy(stops[ns], cand[i].p, 12); core[ns++] = 1; }
    }
    if (N.kind != 2) {        /* the cluster itself */
        int at = (int)(frand() * (ns + 1)) % (ns + 1);
        for (int i = ns; i > at; i--) { memcpy(stops[i], stops[i - 1], 12); core[i] = core[i - 1]; }
        memcpy(stops[at], N.cluster, 12); core[at] = 0; ns++;
    }
    if (ns == 0) { memset(stops[0], 0, 12); core[0] = 0; ns = 1; }
    for (int i = 1; i < ns; i++) {    /* nearest-neighbour order */
        int best = i; float bd = 1e9;
        for (int j = i; j < ns; j++) {
            float dx = stops[j][0]-stops[i-1][0], dy = stops[j][1]-stops[i-1][1], dz = stops[j][2]-stops[i-1][2], d = dx*dx + dy*dy + dz*dz;
            if (d < bd) { bd = d; best = j; }
        }
        float t[3]; memcpy(t, stops[i], 12); memcpy(stops[i], stops[best], 12); memcpy(stops[best], t, 12);
        int c = core[i]; core[i] = core[best]; core[best] = c;
    }
    nwp = 1;
    for (int i = 0; i < ns; i++) { pick_view(stops[i], core[i] ? 0.4f + 0.08f * frand() : 0.7f, wp[nwp]); memcpy(wpT[nwp], stops[i], 12); nwp++; }
    entry_point(wp[1], 1.65f, wp[0]);
    entry_point(wp[nwp - 1], 1.8f, wp[nwp]);
    nwp++;
    memset(wpT[0], 0, 12); memset(wpT[nwp - 1], 0, 12);

    tourDur = tourOpt;
    float seg = tourDur / (nwp - 1);
    for (int i = 0; i < nwp; i++) segStart[i] = i * seg;
    for (int i = 0; i < ns; i++) if (core[i]) add_proto(stops[i], segStart[i + 1] - 7 + 2 * frand());
    /* a few more cores igniting off in the distance */
    for (int i = 0, n = 0; i < nc && n < 6; i++)
        if (!near_proto(cand[i].p, 0.28f)) { add_proto(cand[i].p, 6 + frand() * (tourDur - 20)); n++; }
    make_field();
}

/* --- continuous journey: an endless chain of stops --- */
static int stopsSinceCluster = 0;
static float nextAmb = 0;

static int cur_segment(float t) { int s = 0; while (s < nwp - 2 && t >= segStart[s + 1]) s++; return s; }

static void add_stop(void)
{
    const float *prev = wpT[nwp - 1];
    float target[3]; int isCore = 1;
    if (N.kind != 2 && stopsSinceCluster >= 4) { memcpy(target, N.cluster, 12); isCore = 0; stopsSinceCluster = 0; }
    else {
        int best = -1; float bs = -1;
        int lim = nc < 400 ? nc : 400;
        for (int i = 0; i < lim; i++) {
            float dx = cand[i].p[0]-prev[0], dy = cand[i].p[1]-prev[1], dz = cand[i].p[2]-prev[2], d = sqrtf(dx*dx + dy*dy + dz*dz);
            float sc = cand[i].s * (d > 0.3f && d < 0.9f ? 1 : 0.15f) * (tn - cand[i].lastUse > 240 ? 1 : 0.02f) * (0.6f + 0.8f * frand());
            if (sc > bs) { bs = sc; best = i; }
        }
        if (best < 0) { memcpy(target, N.cluster, 12); isCore = 0; }
        else { memcpy(target, cand[best].p, 12); cand[best].lastUse = tn; stopsSinceCluster++; }
    }
    if (nwp >= MAXW) return;
    pick_view(target, isCore ? 0.4f + 0.08f * frand() : 0.7f, wp[nwp]);
    memcpy(wpT[nwp], target, 12);
    segStart[nwp] = segStart[nwp - 1] + 30 * (0.85f + 0.3f * frand());
    if (isCore && !near_proto(target, 0.08f)) add_proto(target, segStart[nwp] - 6 + 2 * frand());
    nwp++;
}

static void ensure_future(void)
{
    int s = cur_segment(tn);
    if (s > 16) {                                   /* slide the window */
        int k = s - 4;
        memmove(wp, wp + k, sizeof(wp[0]) * (nwp - k)); memmove(wpT, wpT + k, sizeof(wpT[0]) * (nwp - k));
        memmove(segStart, segStart + k, sizeof(float) * (nwp - k)); nwp -= k; s -= k;
    }
    while (nwp - s < 5 && nwp < MAXW) add_stop();
}

static void begin_journey(void)
{
    find_cands();
    nps = psNext = 0; stopsSinceCluster = 0;
    nwp = 1; segStart[0] = tn; memset(wpT[0], 0, 12);
    add_stop();
    entry_point(wp[1], 1.65f, wp[0]);
    segStart[1] = tn + 34;
    for (int i = 0; i < nps; i++) ps[i].tIgn = segStart[1] - 6;
    ensure_future();
    nextAmb = tn + 10;
    make_field();
}

/* after a nebula swap: keep the path we are on, then continue in the new nebula */
static void continue_journey(void)
{
    find_cands();
    nps = psNext = 0; stopsSinceCluster = 0;
    int s = cur_segment(tn);
    if (nwp > s + 3) nwp = s + 3;
    ensure_future();
    nextAmb = tn + 8;
    make_field();
}

static void ambient_ignition(const float *camPos, const float *camFwd)
{
    for (int tries = 0; tries < 60; tries++) {
        int i = (int)(frand() * (nc < 600 ? nc : 600));
        if (i >= nc) continue;
        float d[3] = { cand[i].p[0]-camPos[0], cand[i].p[1]-camPos[1], cand[i].p[2]-camPos[2] }, L = len3(d);
        if (L < 0.2f || L > 1.0f || dot3(d, camFwd) / L < 0.25f || near_proto(cand[i].p, 0.12f)) continue;
        add_proto(cand[i].p, tn + 1.5f);
        return;
    }
}

static void catmull(const float *p0, const float *p1, const float *p2, const float *p3, float t, float *o)
{
    float t2 = t * t, t3 = t2 * t;
    for (int j = 0; j < 3; j++)
        o[j] = 0.5f * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2*p0[j] - 5*p1[j] + 4*p2[j] - p3[j]) * t2 + (-p0[j] + 3*p1[j] - 3*p2[j] + p3[j]) * t3);
}

static void tour_at(float t, float *eye, float *tgt)
{
    t = clampf(t, segStart[0], segStart[nwp - 1] - 1e-3f);
    int s = cur_segment(t);
    float u = (t - segStart[s]) / (segStart[s + 1] - segStart[s]);
    float ue = u - 0.75f * sinf(2 * PI * u) / (2 * PI);          /* slow down near each stop */
    int a = s > 0 ? s - 1 : 0, d = s + 2 < nwp ? s + 2 : nwp - 1;
    catmull(wp[a], wp[s], wp[s + 1], wp[d], ue, eye);
    float f = smooth(0.1f, 0.8f, u);
    for (int j = 0; j < 3; j++) tgt[j] = lerp(wpT[s][j], wpT[s + 1][j], f);
}

/* ------------------------------------------------------------------ */
/* Camera and sprites                                                  */
/* ------------------------------------------------------------------ */
static float V[16], Pm[16], camE[3], camR[3], camU[3], camF[3], tanX, tanY, camTgt[3];
static int camInit = 0;
static float nebAlpha = 1;           /* nebula visibility (continuous mode fades it between nebulae) */

static void set_camera(const float *eye, const float *tgt, float fovdeg)
{
    float f[3] = { tgt[0]-eye[0], tgt[1]-eye[1], tgt[2]-eye[2] }, up[3] = { 0, 1, 0 }, s[3], u[3];
    norm3(f);
    if (fabsf(f[1]) > 0.98f) { up[0] = 1; up[1] = 0; }
    cross3(f, up, s); norm3(s); cross3(s, f, u);
    memset(V, 0, sizeof V);
    V[0] = s[0]; V[4] = s[1]; V[8] = s[2]; V[1] = u[0]; V[5] = u[1]; V[9] = u[2];
    V[2] = -f[0]; V[6] = -f[1]; V[10] = -f[2];
    V[12] = -dot3(s, eye); V[13] = -dot3(u, eye); V[14] = dot3(f, eye); V[15] = 1;
    memcpy(camR, s, 12); memcpy(camU, u, 12); memcpy(camF, f, 12); memcpy(camE, eye, 12);
    float n = 0.001f, fa = 200;
    tanY = tanf(fovdeg * PI / 360); tanX = tanY * scrW / scrH;
    memset(Pm, 0, sizeof Pm);
    Pm[0] = 1 / tanX; Pm[5] = 1 / tanY; Pm[10] = -(fa + n) / (fa - n); Pm[11] = -1; Pm[14] = -2 * fa * n / (fa - n);
}

typedef struct { float *d; int n, cap; GLuint vbo; } Batch;
static Batch bg, fg;

static void batch_init(Batch *b, int cap) { b->cap = cap; b->n = 0; b->d = malloc(sizeof(float) * 9 * cap); glGenBuffers(1, &b->vbo); }

/* a point light of a given on-screen size in pixels */
static void star_px(Batch *b, const float *p, float r, float g, float bl, float I, float px, float spike)
{
    if (b == &fg) I *= nebAlpha;       /* things inside the nebula fade with it */
    if (b->n >= b->cap || I < 0.003f) return;
    float depth = -(V[2]*p[0] + V[6]*p[1] + V[10]*p[2] + V[14]);
    if (depth < 0.003f) return;
    float vx = V[0]*p[0] + V[4]*p[1] + V[8]*p[2] + V[12], vy = V[1]*p[0] + V[5]*p[1] + V[9]*p[2] + V[13];
    if (fabsf(vx) > depth * tanX * 1.2f || fabsf(vy) > depth * tanY * 1.2f) return;
    float size = depth * px * 5 * (2 * tanY / scrH);
    float *d = b->d + 9 * b->n++;
    d[0] = p[0]; d[1] = p[1]; d[2] = p[2]; d[3] = size; d[4] = r; d[5] = g; d[6] = bl; d[7] = I; d[8] = spike;
}

static void draw_batch(Batch *b, float nebK)
{
    if (!b->n) return;
    glBindBuffer(GL_ARRAY_BUFFER, b->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 9 * b->n, b->d, GL_STREAM_DRAW);
    glUseProgram(pSprite);
    glUniformMatrix4fv(U(pSprite, "V"), 1, GL_FALSE, V); glUniformMatrix4fv(U(pSprite, "P"), 1, GL_FALSE, Pm);
    glUniform1f(U(pSprite, "nebK"), nebK); glUniform2f(U(pSprite, "scr"), scrW, scrH);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, texNeb); glUniform1i(U(pSprite, "nebT"), 3);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vaoSprite);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 36, (void *)0);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 36, (void *)16);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 36, (void *)32);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, b->n);
    b->n = 0;
}

#define NBG 5000
static float bgStar[NBG][7];
static void init_bg(void)
{
    for (int i = 0; i < NBG; i++) {
        rand_unit(bgStar[i]); float t = frand();
        bgStar[i][3] = t < 0.3f ? 1.0f : t < 0.7f ? 0.9f : 0.75f;
        bgStar[i][4] = t < 0.3f ? 0.8f : t < 0.7f ? 0.92f : 0.85f;
        bgStar[i][5] = t < 0.3f ? 0.6f : 1.0f;
        bgStar[i][6] = powf(frand(), 9);
    }
}

/* ------------------------------------------------------------------ */
/* Per-frame scene                                                     */
/* ------------------------------------------------------------------ */
#define NDYN 5
static float dynL[NDYN][4];          /* event lights: pos + intensity */

static float ignite_curve(float e) { return e < 0 ? 0 : smooth(0, 0.5f, e) * 2.5f * expf(-e / 1.8f) + smooth(0, 4, e); }

static void protostar(const Proto *P)
{
    float e = tn - P->tIgn;
    if (e < -25) return;
    float tau = tau_seg(camE, P->p, 24), T = expf(-tau);
    if (e < 0) {     /* a deeply embedded, warming core */
        float w = smooth(-25, 0, e);
        star_px(&fg, P->p, 1, 0.35f, 0.15f, 0.6f * w * fmaxf(T, 0.05f), 5, 0);
        return;
    }
    float I = ignite_curve(e);
    float Ts = fmaxf(T, 0.12f);         /* its own light partly clears a window */
    star_px(&fg, P->p, 1, 0.9f, 0.8f, 3.0f * I * Ts, 3.2f, 1.0f);
    star_px(&fg, P->p, 1, 0.55f, 0.35f, 0.5f * I * Ts, 16, 0);
    if (e > 120) return;                /* old outflows have dispersed */
    /* bipolar jets: irregular knots ploughing into the gas, with a glowing bow shock at the head */
    float jf = smooth(120, 80, e);
    for (int s = -1; s <= 1; s += 2) {
        float ax[3], perp[3], up[3] = { 0, 1, 0 };
        for (int j = 0; j < 3; j++) ax[j] = P->axis[j] * s;
        cross3(ax, up, perp); norm3(perp);
        float head = 0.006f + 0.0085f * fmaxf(e - 1.2f, 0);
        if (head > 0.13f) head = 0.13f;
        for (int k = 0; k < 110; k++) {
            float h = hash1(k * 13 + s * 7 + P->id * 101), age = e - 1.2f - k * 0.32f - h * 0.2f;
            if (age < 0) continue;
            float dist = fminf(0.006f + 0.0085f * age * (0.85f + 0.3f * h), head);
            float wob = 0.008f * dist / 0.13f * sinf(k * 0.9f + h * 6) + (0.002f + 0.03f * dist) * (hash1(k * 31 + s) - 0.5f);
            float q[3];
            for (int j = 0; j < 3; j++) q[j] = P->p[j] + ax[j] * dist + perp[j] * wob;
            float hk = hash1(k * 5 + s), b = (0.2f + (hk > 0.9f ? 1.2f : 0.3f * hk)) * expf(-age / 22) * jf, Tq = fmaxf(expf(-tau_seg(camE, q, 10)), 0.06f);
            star_px(&fg, q, 1, 0.38f + 0.2f * h, 0.3f, 0.14f * b * Tq, 3 + 4 * h, 0);
        }
        if (e > 1.2f) {
            float q[3]; for (int j = 0; j < 3; j++) q[j] = P->p[j] + ax[j] * head;
            float Tq = fmaxf(expf(-tau_seg(camE, q, 12)), 0.06f), b = smooth(1.2f, 5, e) * (0.8f + 0.2f * sinf(tn * 2 + s)) * jf;
            star_px(&fg, q, 1, 0.45f, 0.35f, 0.18f * b * Tq, 14, 0);
            star_px(&fg, q, 0.5f, 0.7f, 1, 0.08f * b * Tq, 30, 0);
        }
    }
}

/* the newest ignitions light the gas; the oldest of five hands over its light gracefully */
static void assign_lights(void)
{
    memset(dynL, 0, sizeof dynL);
    int idx[MAXPS], n = 0;
    for (int i = 0; i < nps; i++) if (ps[i].tIgn <= tn) idx[n++] = i;
    for (int i = 1; i < n; i++) { int v = idx[i], j = i - 1; while (j >= 0 && ps[idx[j]].tIgn < ps[v].tIgn) { idx[j + 1] = idx[j]; j--; } idx[j + 1] = v; }
    float newestAge = n ? tn - ps[idx[0]].tIgn : 0;
    for (int k = 0; k < n && k < NDYN; k++) {
        const Proto *P = &ps[idx[k]];
        float w = k < NDYN - 1 ? 1 : 1 - smooth(0, 6, newestAge);
        memcpy(dynL[k], P->p, 12); dynL[k][3] = 0.0004f * ignite_curve(tn - P->tIgn) * w * nebAlpha;
    }
}

static void build_frame(float dt)
{
    float eye[3], tgt[3];
    tour_at(tn, eye, tgt);
    if (!camInit) { memcpy(camTgt, tgt, 12); camInit = 1; }
    float k = 1 - expf(-dt / 1.6f);
    for (int j = 0; j < 3; j++) camTgt[j] += (tgt[j] - camTgt[j]) * k;
    set_camera(eye, camTgt, 58);

    for (int i = 0; i < NBG; i++) {
        float p[3] = { camE[0] + bgStar[i][0] * 50, camE[1] + bgStar[i][1] * 50, camE[2] + bgStar[i][2] * 50 }, m = bgStar[i][6];
        star_px(&bg, p, bgStar[i][3], bgStar[i][4], bgStar[i][5], 0.12f + 2.0f * m, 1.1f + 1.5f * m, m > 0.5f ? 0.35f : 0);
    }
    if (nebAlpha <= 0.002f) { memset(dynL, 0, sizeof dynL); return; }
    for (int i = 0; i < NFIELD; i++) {
        float T = expf(-tau_seg(camE, field[i], 16));
        star_px(&fg, field[i], field[i][3], field[i][4], field[i][5], field[i][6] * T, 1.4f + field[i][6] * 0.6f, field[i][6] > 1.5f ? 0.5f : 0);
    }
    for (int i = 0; i < N.nL; i++) {     /* the hot stars powering the nebula */
        float T = fmaxf(expf(-tau_seg(camE, N.lp[i], 24)), 0.03f), c = fmaxf(N.lc[i][0], fmaxf(N.lc[i][1], N.lc[i][2]));
        float tw = 1 + 0.04f * sinf(tn * 3.1f + i * 2.3f);
        star_px(&fg, N.lp[i], N.lc[i][0] / c, N.lc[i][1] / c, N.lc[i][2] / c, 7 * T * tw, 3.5f, 1.0f);
        star_px(&fg, N.lp[i], N.lc[i][0] / c, N.lc[i][1] / c, N.lc[i][2] / c, 0.7f * T, 22, 0);
    }
    for (int i = 0; i < nps; i++) protostar(&ps[i]);
    assign_lights();
}

static float postFade = 1;

static void render(int timeQuery, int parity)
{
    ensure_targets();
    int q = tqFrame & 1;
    if (timeQuery) glBeginQuery(GL_TIME_ELAPSED, tq[q]);
    /* 1. ray-march the nebula at the current quality level (not every frame on slow GPUs) */
    glDisable(GL_BLEND);
    int visible = nebAlpha > 0.002f;
    if (!visible) nebFresh = 1;
    else {
        if (nebFresh) parity = -1;           /* a fresh target needs every row */
        nebFresh = 0;
        glBindFramebuffer(GL_FRAMEBUFFER, fboNeb);
        glViewport(0, 0, nebW, nebH);
        glUseProgram(pMarch);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, volTex);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, detailTex);
        glUniform1i(U(pMarch, "vol"), 0); glUniform1i(U(pMarch, "detail"), 2);
        glUniform3f(U(pMarch, "lightTint"), 0.85f, 0.92f, 1.0f);
        glUniform1f(U(pMarch, "pixAng"), 2 * tanY / nebH);
        glUniform3fv(U(pMarch, "camPos"), 1, camE); glUniform3fv(U(pMarch, "camR"), 1, camR);
        glUniform3fv(U(pMarch, "camU"), 1, camU); glUniform3fv(U(pMarch, "camF"), 1, camF);
        glUniform1f(U(pMarch, "tanX"), tanX); glUniform1f(U(pMarch, "tanY"), tanY);
        glUniform1i(U(pMarch, "steps"), qSteps[qLevel]); glUniform1f(U(pMarch, "time"), fmodf(tn, 2000.0f));
        glUniform3fv(U(pMarch, "colHa"), 1, N.colHa); glUniform3fv(U(pMarch, "colO3"), 1, N.colO3); glUniform3fv(U(pMarch, "colRefl"), 1, N.colRefl);
        glUniform1f(U(pMarch, "kD"), KD); glUniform1f(U(pMarch, "kG"), KG);
        glUniform1f(U(pMarch, "emGain"), N.emGain); glUniform1f(U(pMarch, "reflGain"), N.reflGain); glUniform1f(U(pMarch, "ambient"), N.ambient);
        glUniform4fv(U(pMarch, "dl"), NDYN, &dynL[0][0]);
        glUniform1i(U(pMarch, "rowParity"), parity);
        fs_draw();
    }

    /* 2. stars at full resolution: background ones dimmed by the gas in front of them,
     *    embedded stars, protostars and jets added on top. The gas itself is added in
     *    the final pass, which saves a full-screen composite pass. */
    float k = visible ? nebAlpha : 0;
    glBindFramebuffer(GL_FRAMEBUFFER, fboHdr);
    glViewport(0, 0, scrW, scrH);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE);
    draw_batch(&bg, k);
    draw_batch(&fg, 0);
    glDisable(GL_BLEND);

    /* 5. bloom at quarter resolution */
    int bw = (scrW + 3) / 4, bh = (scrH + 3) / 4;
    glViewport(0, 0, bw, bh);
    glBindFramebuffer(GL_FRAMEBUFFER, fboB[0]);
    glActiveTexture(GL_TEXTURE0);
    glUseProgram(pBright); glBindTexture(GL_TEXTURE_2D, texHdr); glUniform1i(U(pBright, "src"), 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texNeb); glUniform1i(U(pBright, "neb"), 1);
    glActiveTexture(GL_TEXTURE0); glUniform1f(U(pBright, "nebK"), k);
    glUniform2f(U(pBright, "texel"), 1.0f / scrW, 1.0f / scrH);
    fs_draw();
    glUseProgram(pBlur); glUniform1i(U(pBlur, "src"), 0);
    for (int pass = 0; pass < 2; pass++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fboB[1]); glBindTexture(GL_TEXTURE_2D, texB[0]);
        glUniform2f(U(pBlur, "dir"), (1.0f + pass) / bw, 0); fs_draw();
        glBindFramebuffer(GL_FRAMEBUFFER, fboB[0]); glBindTexture(GL_TEXTURE_2D, texB[1]);
        glUniform2f(U(pBlur, "dir"), 0, (1.0f + pass) / bh); fs_draw();
    }

    /* 6. tone map to the screen */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, scrW, scrH);
    glUseProgram(pPost);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texB[0]); glUniform1i(U(pPost, "bloom"), 1);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texHdr); glUniform1i(U(pPost, "hdr"), 0);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, texNeb); glUniform1i(U(pPost, "neb"), 2);
    glActiveTexture(GL_TEXTURE0);
    glUniform1f(U(pPost, "nebK"), k); glUniform2f(U(pPost, "nebTexel"), 1.0f / nebW, 1.0f / nebH);
    glUniform1f(U(pPost, "exposure"), N.exposure);
    glUniform1f(U(pPost, "fade"), postFade);
    fs_draw();
    if (timeQuery) { glEndQuery(GL_TIME_ELAPSED); tqUsed[q] = 1; tqFrame++; }
}

/* adjust ray-march resolution from the measured GPU time of that pass */
static void auto_quality(double t)
{
    int q = tqFrame & 1;          /* the query issued before the latest one */
    if (tqUsed[q]) {
        GLint ready = 0; glGetQueryObjectiv(tq[q], GL_QUERY_RESULT_AVAILABLE, &ready);
        if (ready) { GLuint64 ns; glGetQueryObjectui64v(tq[q], GL_QUERY_RESULT, &ns); marchMs = marchMs * 0.85f + (ns / 1e6f) * 0.15f; tqUsed[q] = 0; }
    }
    if (!qAuto || t - qLastChange < 2.5 || marchMs <= 0) return;
    /* whole-frame GPU time against a 60 Hz frame (16.7 ms), with headroom */
    if (marchMs > 14.5f && qLevel > 0) { qLevel--; qLastChange = t; marchMs = 0; }
    else if (marchMs < 9.5f && qLevel < NQ - 1) { qLevel++; qLastChange = t; marchMs = 0; }
}

static void seed_rng(unsigned int seed)
{
    rng = seed * 2654435761u ^ 0x5bd1e995u; rng ^= rng >> 15; rng *= 2246822519u; rng ^= rng >> 13; if (!rng) rng = 1;
    for (int i = 0; i < 8; i++) frand();
}

/* classic mode: a fresh nebula and tour, faded in from black */
static void new_nebula(unsigned int seed)
{
    unsigned int keep = rng;
    seed_rng(seed);
    design_nebula();
    gen_begin(); gen_step(1);
    plan_tour();
    rng = keep ^ (seed * 2654435761u);
    if (!rng) rng = 1;
    tn = 0; camInit = 0; nebFresh = 1;
}

/* continuous mode: the nebula fades out around the moving camera, the next one is
 * built over a few dozen frames, then it fades in - the flight never stops */
enum { SHOW, FADE_OUT, BUILD, FADE_IN };
static int phase = SHOW;
static float phaseT = 0, nebStart = 0;
#define XFADE 6.0f

static void first_continuous(unsigned int seed)
{
    seed_rng(seed);
    design_nebula();
    gen_begin(); gen_step(1);
    tn = 0; begin_journey();
    phase = SHOW; nebStart = 0; nebAlpha = 1; camInit = 0; nebFresh = 1;
}

static void request_switch(void) { if (phase == SHOW) { phase = FADE_OUT; phaseT = tn; } }

static void continuous_tick(int sync, const float *camPos, const float *camFwd)
{
    ensure_future();
    switch (phase) {
    case SHOW:
        nebAlpha = 1;
        if (tn - nebStart >= tourDur) request_switch();
        if (tn >= nextAmb) { ambient_ignition(camPos, camFwd); nextAmb = tn + 14 + 12 * frand(); }
        break;
    case FADE_OUT:
        nebAlpha = 1 - smooth(0, XFADE, tn - phaseT);
        if (tn - phaseT >= XFADE) { phase = BUILD; nebAlpha = 0; design_nebula(); gen_begin(); }
        break;
    case BUILD:
        nebAlpha = 0;
        if (gen_step(sync)) { continue_journey(); phase = FADE_IN; phaseT = tn; nebStart = tn; nebFresh = 1; }
        break;
    case FADE_IN:
        nebAlpha = smooth(0, XFADE, tn - phaseT);
        if (tn - phaseT >= XFADE) phase = SHOW;
        break;
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int windowed = 0, rootmode = 0, showfps = 0, shot = 0;
    float speed = 1, shotT = 0; unsigned int shotSeed = 1; const char *shotFile = NULL; Window target = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == '-') a++;
        int more = i + 1 < argc;
        if (!strcmp(a, "-window")) windowed = 1;
        else if (!strcmp(a, "-root")) rootmode = 1;
        else if (!strcmp(a, "-window-id") && more) target = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "-fps")) showfps = 1;
        else if (!strcmp(a, "-speed") && more) speed = clampf(atof(argv[++i]), 0.05f, 10);
        else if (!strcmp(a, "-duration") && more) { tourOpt = clampf(atof(argv[++i]), 40, 3600); durGiven = 1; }
        else if (!strcmp(a, "-continuous")) continuous = 1;
        else if (!strcmp(a, "-palette") && more) { const char *v = argv[++i]; paletteOpt = !strcmp(v, "natural") ? 0 : !strcmp(v, "hubble") ? 1 : -1; }
        else if (!strcmp(a, "-quality") && more) {
            const char *v = argv[++i]; qAuto = 0;
            if (!strcmp(v, "low")) qLevel = 0; else if (!strcmp(v, "medium")) qLevel = 2;
            else if (!strcmp(v, "high")) qLevel = 4; else if (!strcmp(v, "ultra")) qLevel = 6;
            else { qAuto = 1; qLevel = 3; }
        }
        else if (!strcmp(a, "-shot") && i + 3 < argc) { shot = 1; shotSeed = strtoul(argv[++i], NULL, 0); shotT = atof(argv[++i]); shotFile = argv[++i]; windowed = 1; }
        else {
            int help = !strcmp(a, "-h") || !strcmp(a, "-help");
            if (!help) fprintf(stderr, "nebulae: unknown option %s\n", argv[i]);
            printf("usage: %s [-window] [-root] [-window-id ID] [-quality auto|low|medium|high|ultra]\n"
                   "       [-palette mixed|natural|hubble] [-duration SECONDS] [-speed X] [-fps]\n"
                   "       [-continuous] [-shot SEED TIME file.ppm]\n"
                   "  -continuous   one endless flight: stars keep forming, nebulae change without a break\n"
                   "  -duration S   seconds per nebula (default 130, or 360 with -continuous)\n", argv[0]);
            return help ? 0 : 1;
        }
    }
    if (continuous && !durGiven) tourOpt = 360;
    tourDur = tourOpt;
    const char *env = getenv("XSCREENSAVER_WINDOW");
    if (!target && env && *env) target = strtoul(env, NULL, 0);

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "nebulae: cannot open display\n"); return 1; }
    int scr = DefaultScreen(dpy);
    Window root = RootWindow(dpy, scr), win;
    if (!target && rootmode) target = root;
    XVisualInfo *vi; int ownwin = !target;
    if (target) {
        XWindowAttributes wa; XGetWindowAttributes(dpy, target, &wa);
        XVisualInfo tmpl; int n; tmpl.visualid = XVisualIDFromVisual(wa.visual);
        vi = XGetVisualInfo(dpy, VisualIDMask, &tmpl, &n);
        win = target;
    } else {
        int attr[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
        vi = glXChooseVisual(dpy, scr, attr);
        if (!vi) { fprintf(stderr, "nebulae: no suitable GLX visual\n"); return 1; }
        XSetWindowAttributes swa; memset(&swa, 0, sizeof swa);
        swa.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone);
        swa.event_mask = KeyPressMask | ButtonPressMask | PointerMotionMask | StructureNotifyMask;
        int w = windowed ? 1280 : DisplayWidth(dpy, scr), h = windowed ? 720 : DisplayHeight(dpy, scr);
        win = XCreateWindow(dpy, root, 0, 0, w, h, 0, vi->depth, InputOutput, vi->visual,
                            CWColormap | CWEventMask | CWBorderPixel | CWBackPixel, &swa);
        XStoreName(dpy, win, "Nebulae");
        Atom del = XInternAtom(dpy, "WM_DELETE_WINDOW", False); XSetWMProtocols(dpy, win, &del, 1);
        if (!windowed) {
            Atom st = XInternAtom(dpy, "_NET_WM_STATE", False), fs = XInternAtom(dpy, "_NET_WM_STATE_FULLSCREEN", False);
            XChangeProperty(dpy, win, st, XA_ATOM, 32, PropModeReplace, (unsigned char *)&fs, 1);
            static const char blank[1] = { 0 }; XColor black; memset(&black, 0, sizeof black);
            Pixmap bm = XCreateBitmapFromData(dpy, win, blank, 1, 1);
            XDefineCursor(dpy, win, XCreatePixmapCursor(dpy, bm, bm, &black, &black, 0, 0));
        }
        if (!shot) XMapRaised(dpy, win);
    }
    GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
    if (!ctx || !glXMakeCurrent(dpy, win, ctx)) { fprintf(stderr, "nebulae: cannot create GL context\n"); return 1; }
    int (*swapInterval)(int) = (int (*)(int))glXGetProcAddressARB((const GLubyte *)"glXSwapIntervalMESA");
    if (swapInterval) swapInterval(1);

    rng ^= (unsigned)time(NULL) * 2654435761u ^ (unsigned)getpid(); if (!rng) rng = 1;
    gl_setup();
    batch_init(&bg, NBG + 16); batch_init(&fg, 8000);
    if (shot) { rng = 0x2545F491u; }
    init_bg();

    if (shot) {
        scrW = 1280; scrH = 720;
        if (qAuto) qLevel = 5;
        float dt = 1 / 30.0f;
        if (continuous) {
            first_continuous(shotSeed);
            while (tn + dt < shotT) { tn += dt; continuous_tick(1, camE, camF); build_frame(dt); fg.n = bg.n = 0; }
            tn = shotT; continuous_tick(1, camE, camF);
            postFade = smooth(0, 4, tn);
        } else {
            new_nebula(shotSeed);
            while (tn + dt < shotT) { tn += dt; build_frame(dt); fg.n = bg.n = 0; }
            tn = shotT;
            postFade = smooth(0, 4, tn) * smooth(tourDur, tourDur - 4, tn);
        }
        build_frame(dt);
        render(0, -1); glFinish();
        unsigned char *px = malloc(scrW * scrH * 3);
        glReadBuffer(GL_BACK); glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, scrW, scrH, GL_RGB, GL_UNSIGNED_BYTE, px);
        FILE *f = fopen(shotFile, "wb");
        if (!f) { perror(shotFile); return 1; }
        fprintf(f, "P6\n%d %d\n255\n", scrW, scrH);
        for (int y = scrH - 1; y >= 0; y--) fwrite(px + y * scrW * 3, 1, scrW * 3, f);
        fclose(f);
        fprintf(stderr, "kind %d palette %d, %d waypoints, %d protostars, nebula alpha %.2f, phase %d\n", N.kind, N.palette, nwp, nps, nebAlpha, phase);
        return 0;
    }

    {   /* window size once; resize events keep it current, so there is no per-frame
         * round trip to the X server (those stalled for up to 100 ms under a compositor) */
        XWindowAttributes wa; XGetWindowAttributes(dpy, win, &wa);
        scrW = wa.width > 0 ? wa.width : 1; scrH = wa.height > 0 ? wa.height : 1;
        if (!ownwin) XSelectInput(dpy, win, StructureNotifyMask);
    }
    if (continuous) first_continuous((unsigned)(frand() * 4e9));
    else new_nebula((unsigned)(frand() * 4e9));
    double last = now(), fpsT = last, startT = last; int frames = 0, mx0 = -1, my0 = -1, frameNo = 0;
    for (;;) {
        while (XPending(dpy)) {
            XEvent ev; XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify) { scrW = ev.xconfigure.width > 0 ? ev.xconfigure.width : 1; scrH = ev.xconfigure.height > 0 ? ev.xconfigure.height : 1; continue; }
            if (!ownwin) continue;
            if (ev.type == ClientMessage) return 0;
            if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (windowed && (k == XK_n || k == XK_space)) { if (continuous) request_switch(); else { new_nebula((unsigned)(frand() * 4e9)); last = now(); } continue; }
                if (!windowed || k == XK_q || k == XK_Escape) return 0;
            }
            if (!windowed && ev.type == ButtonPress) return 0;
            if (!windowed && ev.type == MotionNotify && now() - startT > 1.0) {
                if (mx0 < 0) { mx0 = ev.xmotion.x_root; my0 = ev.xmotion.y_root; }
                else if (abs(ev.xmotion.x_root - mx0) + abs(ev.xmotion.y_root - my0) > 20) return 0;
            }
        }
        double t = now(); float dt = (float)(t - last); last = t;
        if (dt > 0.1f) dt = 0.1f;
        tn += dt * speed;
        if (continuous) {
            continuous_tick(0, camE, camF);
            postFade = smooth(0, 4, tn);
        } else {
            if (tn >= tourDur) { new_nebula((unsigned)(frand() * 4e9)); last = now(); continue; }
            postFade = smooth(0, 4, tn) * smooth(tourDur, tourDur - 4, tn);
        }
        build_frame(dt * speed);
        frameNo++;
        render(1, qEvery[qLevel] > 1 ? (frameNo & 1) : -1);
        glXSwapBuffers(dpy, win);
        auto_quality(t);

        frames++;
        if (showfps && t - fpsT > 2) {
            fprintf(stderr, "t=%.0f  %.1f fps  quality %d (%.0f%% res, %d steps)  gpu %.1f ms  phase %d  protostars %d\n",
                    tn, frames / (t - fpsT), qLevel, qScale[qLevel] * 100, qSteps[qLevel], marchMs, phase, nps);
            frames = 0; fpsT = t;
        }
    }
}
