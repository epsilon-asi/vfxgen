"""Basic light atoms shared by many effects: glow, flash, shockwave, ring."""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, atom
from ..engine.mathx import TAU, smoothstep
from ..engine.noise import Noise


def _end_fade(t: float, duration: float) -> float:
    """1 until 80% of the duration, then smoothly 0 at the end (one-shots end empty)."""
    return float(smoothstep(duration, duration * 0.8, t))


@atom("glow", tags=["light", "core", "halo"])
class Glow(Atom):
    """Steady soft glow: a gaussian core plus optional halo, with optional pulse and flicker.

    Use for orb cores, fire base glow, torch halos, emitter flares, highlights."""
    kind = "loop"
    outputs = {"light": Light(palette="fire")}
    streams = ("flicker",)

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        radius: float = P(10.0, gt=0, doc="core radius (gaussian sigma, px)")
        brightness: float = P(2.0, ge=0, doc="core peak intensity")
        halo_radius: float = P(26.0, ge=0, doc="halo radius (sigma, px); 0 = no halo")
        halo_brightness: float = P(0.5, ge=0, doc="halo peak intensity")
        squash: float = P(1.0, gt=0, doc="vertical scale of the glow (ellipse)")
        pulse: float = P(0.0, ge=0, le=1, doc="pulse depth (0..1)")
        pulse_cycles: int = P(1, ge=0, doc="pulses per period (whole number: loops stay seamless)")
        flicker: float = P(0.0, ge=0, le=1, doc="random flicker depth (0..1)")
        flicker_rate: float = P(12.0, gt=0, doc="flicker changes per second")

    def get_bounds(self):
        r = 3.2 * max(self.p.radius, self.p.halo_radius)
        return (-r, -r * self.p.squash, r, r * self.p.squash)

    def draw(self, ctx, t):
        p = self.p
        k = 1.0
        if p.pulse:
            k *= 1 + p.pulse * math.sin(TAU * p.pulse_cycles * t / p.period)
        if p.flicker:
            k *= 1 - p.flicker * float(self.tick_rng("flicker", t, p.flicker_rate).random())
        if p.squash == 1.0:
            ctx.light.dot(0, 0, p.radius * (1 + 0.3 * (k - 1)), p.brightness * k)
            if p.halo_radius > 0:
                ctx.light.dot(0, 0, p.halo_radius * (1 + 0.3 * (k - 1)), p.halo_brightness * k)
        else:
            sq = p.squash

            def f(X, Y):
                d2 = X * X + (Y / sq) ** 2
                v = p.brightness * k * np.exp(-d2 / (2 * p.radius ** 2))
                if p.halo_radius > 0:
                    v = v + p.halo_brightness * k * np.exp(-d2 / (2 * p.halo_radius ** 2))
                return v
            ctx.light.field(f, self.get_bounds())


@atom("flash", tags=["light", "burst"])
class Flash(Atom):
    """Bright burst that grows and decays fast, with an optional slower afterglow."""
    kind = "oneshot"
    outputs = {"light": Light(palette="fire")}

    class Params(Atom.Params):
        duration: float = P(0.4, gt=0, doc="seconds until fully gone")
        radius: float = P(10.0, gt=0, doc="starting radius (sigma, px)")
        grow: float = P(80.0, ge=0, doc="radius added per second")
        brightness: float = P(3.2, ge=0, doc="peak intensity")
        decay: float = P(0.05, gt=0, doc="brightness half-life-ish time constant (s)")
        afterglow_radius: float = P(32.0, ge=0, doc="afterglow radius (px); 0 = none")
        afterglow_brightness: float = P(0.6, ge=0, doc="afterglow peak intensity")
        afterglow_decay: float = P(0.12, gt=0, doc="afterglow time constant (s)")

    def get_bounds(self):
        r = 3.2 * max(self.p.radius + self.p.grow * self.p.duration, self.p.afterglow_radius)
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        e = _end_fade(t, p.duration)
        ctx.light.dot(0, 0, p.radius + p.grow * t, p.brightness * math.exp(-t / p.decay) * e)
        if p.afterglow_radius > 0:
            ctx.light.dot(0, 0, p.afterglow_radius, p.afterglow_brightness * math.exp(-t / p.afterglow_decay) * e)


@atom("shockwave", tags=["light", "ring", "burst"])
class Shockwave(Atom):
    """Expanding ring that fades as it grows. Optional noise breakup and inner haze.

    Radius follows r0 + (r1 - r0) * (1 - exp(-t / expand_time)); brightness decays with
    fade_time. ``squash`` < 1 makes a ground-plane ellipse."""
    kind = "oneshot"
    outputs = {"light": Light(palette="fire")}
    streams = ("noise",)

    class Params(Atom.Params):
        duration: float = P(0.35, gt=0, doc="seconds until fully gone")
        radius_start: float = P(12.0, ge=0, doc="initial radius (px)")
        radius_end: float = P(104.0, ge=0, doc="radius it approaches (px)")
        expand_time: float = P(0.07, gt=0, doc="expansion time constant (s)")
        width_start: float = P(4.4, gt=0, doc="ring thickness at start (px)")
        width_end: float = P(4.4, gt=0, doc="ring thickness at the end (px)")
        brightness: float = P(0.9, ge=0, doc="peak intensity")
        fade_time: float = P(0.09, gt=0, doc="brightness time constant (s)")
        squash: float = P(1.0, gt=0, le=1, doc="vertical squash (ground ellipse < 1)")
        breakup: float = P(0.0, ge=0, le=1, doc="noise breakup of the ring (0..1)")
        breakup_scale: float = P(3.0, gt=0, doc="noise frequency around the ring")
        haze: float = P(0.0, ge=0, doc="inner fill brightness")

    def setup(self):
        self.noise = Noise(self.rng("noise"), period=(16, 16, 4))

    def get_bounds(self):
        r = max(self.p.radius_end, self.p.radius_start) + 3 * max(self.p.width_start, self.p.width_end) + 4
        return (-r, -r * self.p.squash - 2, r, r * self.p.squash + 2)

    def draw(self, ctx, t):
        p = self.p
        q = min(t / p.duration, 1.0)
        R = p.radius_start + (p.radius_end - p.radius_start) * (1 - math.exp(-t / p.expand_time))
        w = p.width_start + (p.width_end - p.width_start) * q
        a = p.brightness * math.exp(-t / p.fade_time) * _end_fade(t, p.duration)
        if a < 1e-4:
            return
        if p.breakup <= 0 and p.haze <= 0:
            ctx.light.ring(0, 0, R, w, a, squash=p.squash)
            return
        sq = p.squash

        def f(X, Y):
            dy = Y / sq
            r = np.hypot(X, dy)
            v = a * np.exp(-((r - R) / w) ** 2 / 2)
            if p.breakup > 0:
                ph = np.arctan2(dy, X)
                n = self.noise.fbm(np.cos(ph) * p.breakup_scale, np.sin(ph) * p.breakup_scale, q * 2, octaves=3)
                v = v * np.clip(1 - p.breakup + p.breakup * (0.6 + 1.6 * n), 0, None)
            if p.haze > 0:
                v = v + a * p.haze * smoothstep(R, R * 0.4, r) * 0.5
            return v
        ctx.light.field(f, self.get_bounds())


@atom("ring", tags=["light", "ring", "static"])
class Ring(Atom):
    """Steady ring: optional dashes that rotate, and a pulse. Ground rings, rune circles."""
    kind = "loop"
    outputs = {"light": Light(palette="arcane")}

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        radius: float = P(90.0, ge=0, doc="ring radius (px)")
        width: float = P(2.0, gt=0, doc="ring thickness (px)")
        brightness: float = P(1.0, ge=0, doc="peak intensity")
        squash: float = P(1.0, gt=0, le=1, doc="vertical squash (ground ellipse < 1)")
        dashes: int = P(0, ge=0, doc="number of dashes (0 = solid)")
        duty: float = P(0.5, gt=0, le=1, doc="fraction of each dash period that is lit")
        rotation: int = P(1, doc="dash steps advanced per period (whole number; negative = reverse)")
        pulse: float = P(0.0, ge=0, le=1, doc="pulse depth")
        pulse_cycles: int = P(1, ge=0, doc="pulses per period")

    def get_bounds(self):
        r = self.p.radius + 3 * self.p.width + 2
        return (-r, -r * self.p.squash - 2, r, r * self.p.squash + 2)

    def draw(self, ctx, t):
        p = self.p
        k = 1 + p.pulse * math.sin(TAU * p.pulse_cycles * t / p.period) if p.pulse else 1.0
        phase = -p.rotation * t / p.period if p.dashes else 0.0
        ctx.light.ring(0, 0, p.radius, p.width, p.brightness * k, squash=p.squash,
                       dashes=p.dashes, duty=p.duty, phase=phase)
