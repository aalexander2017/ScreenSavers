# ScreenSavers

Two OpenGL screensavers for [XScreenSaver](https://www.jwz.org/xscreensaver/) on Linux/X11.

## Cosmogenesis

A tour through cosmic history, looping:

1. The Big Bang: fireball, cooling, dark ages, the cosmic web
2. A spiral galaxy forms
3. Two galaxies collide and merge (restricted N-body)
4. The merged galaxy's supermassive black hole: accretion disk, gravitational
   lensing, jets, and a star torn apart in a tidal disruption event
5. A stellar nursery: clouds collapse and stars ignite
6. A protoplanetary disk becomes a planetary system
7. A red supergiant goes supernova and leaves a pulsar

Options: `--speed`, `--scene N` (or `-1` for random), `--shuffle`,
`--no-bigbang`, `--no-galaxy`, ... to leave scenes out. Run with `--help`.

## Nebulae

Slow fly-throughs of ray-marched volumetric nebulae. Each one is generated
fresh: a cluster-carved cavity with dust pillars, a wind-blown bubble, or a
dark dust cloud with blue reflection nebulosity. Starlight is shadowed by the
dust; protostars ignite in dense cores as the camera arrives, light the gas
around them and launch jets.

- `--continuous`: one endless flight; stars keep forming and nebulae change
  without a break
- `--quality auto|low|medium|high|ultra`: auto adapts to the GPU by measuring
  frame time
- `--palette mixed|natural|hubble`, `--duration S`, `--speed X`

Both run full screen on their own, in a window with `--window`
(n / space = next, q = quit), or inside XScreenSaver.

## Build and install

Needs OpenGL 3.3 and the X11/GL development headers
(`sudo apt install libgl-dev libx11-dev` on Debian/Ubuntu).

```sh
make            # build both
make install    # binaries to ~/.local/bin, settings panels to /usr/share/xscreensaver/config (sudo)
```

Then add them to the `programs:` list in `~/.xscreensaver`:

```
  GL:  "Cosmogenesis"  ~/.local/bin/cosmogenesis --root  \n\
  GL:  "Nebulae"       ~/.local/bin/nebulae --root       \n\
```

and pick them in `xscreensaver-settings`.
