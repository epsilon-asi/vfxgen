"""Generic particle atoms. Many effects need only YAML on top of these:
sparks, embers, motes, dust kicks, inflow, glitter, rain of plus-signs..."""
from __future__ import annotations

from ..atom import Atom, Light, P, Smoke, Volume, atom
from ..engine.particles import DIRECTIONS, SHAPES, Emitter, EmitterConfig, draw_particles

_STYLES = ["dot", "streak", "glint", "plus", "puff"]
_OUT = ["light", "smoke", "volume"]


class _ParticleParams(Atom.Params):
    count: int = P(60, ge=0, doc="number of particles")
    shape: str = P("point", doc="spawn shape", choices=list(SHAPES))
    radius: float = P(0.0, ge=0, doc="spawn shape size: circle/disk/ellipse radius, line half-length, box half-width (px)")
    radius_y: float = P(-1.0, doc="ellipse y radius / box half-height (px); -1 = same as radius")
    line_angle: float = P(0.0, doc="line spawn shape angle (deg)")
    direction: str = P("angle", doc="initial direction mode", choices=list(DIRECTIONS))
    angle: float = P(-90.0, doc="emission angle (deg, clockwise from +x; -90 = up)")
    spread: float = P(360.0, ge=0, le=360, doc="emission cone width (deg)")
    speed_min: float = P(40.0, doc="minimum initial speed (px/s)")
    speed_max: float = P(140.0, doc="maximum initial speed (px/s)")
    gravity: float = P(0.0, doc="gravity (px/s², +down; negative = rises)")
    drag: float = P(0.0, ge=0, doc="linear drag (1/s)")
    swirl: float = P(0.0, doc="rotation about the origin (deg/s)")
    sway: float = P(0.0, ge=0, doc="sideways sway amplitude (px)")
    sway_freq: float = P(1.0, ge=0, doc="sway frequency (Hz; loops: per lap)")
    style: str = P("dot", doc="how each particle is drawn", choices=_STYLES)
    size: float = P(1.4, gt=0, doc="particle size (px)")
    size_end: float = P(-1.0, doc="size at end of life (px); -1 = constant")
    brightness: float = P(1.2, ge=0, doc="intensity (light) or density (smoke/volume)")
    brightness_var: float = P(0.4, ge=0, le=1, doc="random brightness variation")
    tail: float = P(0.05, ge=0, doc="streak length (s of travel)")
    fade_in: float = P(0.08, ge=0, le=1, doc="fade-in fraction of life")
    fade_out: float = P(1.5, ge=0, doc="fade-out power (higher = fades sooner)")
    twinkle: float = P(0.0, ge=0, le=1, doc="twinkle depth")
    twinkle_rate: float = P(4.0, ge=0, doc="twinkles per life (loops) or per second")
    halo: float = P(0.0, ge=0, doc="soft halo brightness around each particle")
    extent: float = P(150.0, gt=0, doc="particles fade out before reaching this distance (px); also the bounds")
    output: str = P("light", doc="which output role to draw into", choices=_OUT)
    curl: float = P(0.0, ge=0, doc="swirling flow strength (px/s); 0 = straight paths")
    curl_scale: float = P(80.0, gt=0, doc="size of the swirling eddies (px)")
    curl_speed: float = P(0.5, ge=0, doc="how fast the flow pattern evolves (one-shots)")


def _cfg(p, loop: bool, period: float = 1.0, life=(0.4, 0.8), stagger: float = 0.0, laps=(1, 1)):
    return EmitterConfig(count=p.count, shape=p.shape, radius=p.radius,
                         radius_y=None if p.radius_y < 0 else p.radius_y, line_angle=p.line_angle,
                         direction=p.direction, angle=p.angle, spread=p.spread,
                         speed=(p.speed_min, p.speed_max), life=life, gravity=p.gravity, drag=p.drag,
                         swirl=p.swirl, sway=p.sway, sway_freq=p.sway_freq, stagger=stagger, loop=loop,
                         period=period, laps=laps, curl=p.curl, curl_scale=p.curl_scale,
                         curl_speed=p.curl_speed)


def _draw(self, ctx, t):
    p = self.p
    view = getattr(ctx, p.output)
    draw_particles(view, self.em, t, style=p.style, size=p.size,
                   size_end=None if p.size_end < 0 else p.size_end, brightness=p.brightness,
                   brightness_var=p.brightness_var, tail=p.tail, fade_in=p.fade_in, fade_out=p.fade_out,
                   twinkle=p.twinkle, twinkle_rate=p.twinkle_rate, halo=p.halo, extent=p.extent)


@atom("particles", tags=["particles", "loop"])
class Particles(Atom):
    """Continuous, seamlessly looping particle emission (embers, motes, rising sparks, snow,
    glitter). Each particle lives ``laps_min..laps_max`` lifetimes per period."""
    kind = "loop"
    outputs = {"light": Light(palette="fire"), "smoke": Smoke(palette="smoke"),
               "volume": Volume(palette="dust")}
    streams = ("particles",)

    class Params(_ParticleParams):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        laps_min: int = P(1, ge=1, doc="minimum lifetimes per period (whole number)")
        laps_max: int = P(2, ge=1, doc="maximum lifetimes per period (whole number)")

    def setup(self):
        p = self.p
        self.em = Emitter(self.rng("particles"),
                          _cfg(p, True, p.period, laps=(p.laps_min, max(p.laps_min, p.laps_max))))

    def get_bounds(self):
        e = self.p.extent
        return (-e, -e, e, e)

    draw = _draw


@atom("burst", tags=["particles", "oneshot"])
class Burst(Atom):
    """One-shot particle burst (sparks, debris specks, glitter, dust kicks, pop sparks).
    Particles spawn over ``stagger`` seconds and live ``life_min..life_max`` seconds."""
    kind = "oneshot"
    outputs = {"light": Light(palette="fire"), "smoke": Smoke(palette="smoke"),
               "volume": Volume(palette="dust")}
    streams = ("particles",)

    class Params(_ParticleParams):
        duration: float = P(0.8, gt=0, doc="total length (s); particles are cut at this time")
        life_min: float = P(0.25, gt=0, doc="minimum particle lifetime (s)")
        life_max: float = P(0.55, gt=0, doc="maximum particle lifetime (s)")
        stagger: float = P(0.0, ge=0, doc="spawn times spread over this many seconds")

    def setup(self):
        p = self.p
        self.em = Emitter(self.rng("particles"),
                          _cfg(p, False, life=(p.life_min, max(p.life_min, p.life_max)), stagger=p.stagger))

    def get_bounds(self):
        e = self.p.extent
        return (-e, -e, e, e)

    def draw(self, ctx, t):
        if t >= self.p.duration:
            return
        _draw(self, ctx, t)
