"""Probe atoms with known geometry for engine tests (names start with '_probe')."""
from __future__ import annotations

from vfxgen.atom import ATOMS, Atom, Light, P, Smoke, atom

if "_probe_marker" not in ATOMS:

    @atom("_probe_marker")
    class ProbeMarker(Atom):
        """A dot at local (dx, dy); bounds +-40. Used to check placement / rotation / flips."""
        kind = "oneshot"
        bounds = (-40, -40, 40, 40)
        outputs = {"light": Light(palette="#ffffff", bloom=[], alpha_gamma=1.0)}
        anchors = {"tip": (30, 0), "moving": lambda self, t: (10 * t, 0)}

        class Params(Atom.Params):
            duration: float = P(1.0, gt=0, doc="s")
            dx: float = P(20.0, doc="dot x")
            dy: float = P(0.0, doc="dot y")
            r: float = P(2.0, gt=0, doc="radius")
            a: float = P(4.0, ge=0, doc="brightness")

        def draw(self, ctx, t):
            if t < self.p.duration * 0.999:
                ctx.light.dot(self.p.dx, self.p.dy, self.p.r, self.p.a)

    @atom("_probe_loop")
    class ProbeLoop(Atom):
        """A dot orbiting once per period (radius 20)."""
        kind = "loop"
        bounds = (-30, -30, 30, 30)
        outputs = {"light": Light(palette="#ffffff", bloom=[], alpha_gamma=1.0)}
        streams = ("jitter",)

        class Params(Atom.Params):
            period: float = P(1.0, gt=0, doc="s")
            flicker_rate: float = P(0.0, ge=0, doc="Hz random jitter")

        def draw(self, ctx, t):
            import math
            a = 6.283185 * t / self.p.period
            j = 0.0
            if self.p.flicker_rate:
                j = float(self.tick_rng("jitter", t, self.p.flicker_rate).random()) * 4
            ctx.light.dot(20 * math.cos(a) + j, 20 * math.sin(a), 2.0, 4.0)

    @atom("_probe_smoke")
    class ProbeSmoke(Atom):
        """A smoke blob (smoke role) for channel tests."""
        kind = "oneshot"
        bounds = (-50, -50, 50, 50)
        outputs = {"smoke": Smoke(palette="smoke", lit_by={"channels": ["light"], "amount": 1.0})}

        class Params(Atom.Params):
            duration: float = P(1.0, gt=0, doc="s")

        def draw(self, ctx, t):
            ctx.smoke.dot(0, 0, 15, 3.0)

if "_probe_link" not in ATOMS:
    from vfxgen.atom import Solid, Volume

    @atom("_probe_link")
    class ProbeLink(Atom):
        """Draws a dot at its `a` target (or followed layer) position."""
        kind = "oneshot"
        bounds = (-200, -200, 200, 200)
        outputs = {"light": Light(palette="#ffffff", bloom=[], alpha_gamma=1.0)}

        class Params(Atom.Params):
            duration: float = P(1.0, gt=0, doc="s")
            use_follow: bool = P(False, doc="draw at follow(0.1) instead of target a")

        def draw(self, ctx, t):
            p = ctx.follow(0.1) if self.p.use_follow else ctx.target("a")
            if p is not None:
                ctx.light.dot(p[0], p[1], 2.0, 4.0)

    @atom("_probe_volume")
    class ProbeVolume(Atom):
        """A big density blob for volume / lighting tests."""
        kind = "oneshot"
        bounds = (-60, -60, 60, 60)
        outputs = {"volume": Volume(palette="#404040,#ffffff", bump=20, shadow=0.0, wrap=0.5)}

        class Params(Atom.Params):
            duration: float = P(1.0, gt=0, doc="s")

        def draw(self, ctx, t):
            ctx.volume.dot(0, 0, 18, 6.0)

    @atom("_probe_sprite")
    class ProbeSprite(Atom):
        """Paints an image asset and stamps its alpha as light."""
        kind = "oneshot"
        bounds = (-40, -40, 40, 40)
        outputs = {"solid": Solid(palette="rock"), "light": Light(palette="#ffffff", bloom=[], alpha_gamma=1.0)}

        class Params(Atom.Params):
            duration: float = P(1.0, gt=0, doc="s")
            image: str = P("sprite.png", doc="image path")
            glow: float = P(0.0, ge=0, doc="stamp alpha as light")

        def setup(self):
            self.img = self.asset(self.p.image)

        def draw(self, ctx, t):
            ctx.solid.image(self.img, 0, 0)
            if self.p.glow:
                ctx.light.stamp(self.img, 0, 0, a=self.p.glow)
