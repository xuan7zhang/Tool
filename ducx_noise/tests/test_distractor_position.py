"""Stage 3 acceptance: controllable distractor count / position / similarity.

Verifies the printable tool-list order for a set of parameters, that 'aligned'
distractors mimic real tools (imitates set + name resembles a real tool), and
that 'shuffle' ordering decouples the distractor from a fixed tail slot.
"""

from ducx_noise import apply_noise
from ducx_noise.config import NoiseConfig, DistractorConfig
from ducx_noise.provenance import SOURCE_DISTRACTOR


def _cfg(**kw):
    d = DistractorConfig(enabled=True, count=kw.pop("count", 2),
                         similarity=kw.pop("similarity", "obvious"),
                         position=kw.pop("position", "tail"),
                         index=kw.pop("index", 0))
    return NoiseConfig(seed=kw.pop("seed", 0), distractor=d, **kw)


def _roles(env):
    return [env.manifest[t.name].source for t in env.tools]


def test_position_head_fixed(base_tools):
    env = apply_noise(base_tools, _cfg(count=2, position="head", tool_order="fixed"))
    roles = _roles(env)
    # both distractors sit at the front
    assert roles[0] == SOURCE_DISTRACTOR and roles[1] == SOURCE_DISTRACTOR
    assert roles[2:] == ["real", "real"]


def test_position_tail_fixed(base_tools):
    env = apply_noise(base_tools, _cfg(count=2, position="tail", tool_order="fixed"))
    roles = _roles(env)
    assert roles[:2] == ["real", "real"]
    assert roles[2] == SOURCE_DISTRACTOR and roles[3] == SOURCE_DISTRACTOR


def test_position_index_controlled(base_tools):
    env = apply_noise(base_tools, _cfg(count=1, position="index", index=1,
                                       tool_order="controlled"))
    roles = _roles(env)
    # single distractor inserted at absolute slot 1
    assert roles[1] == SOURCE_DISTRACTOR
    assert roles[0] == "real" and roles[2] == "real"


def test_shuffle_decouples_position(base_tools):
    # Under 'shuffle', the distractor must NOT be pinned to the tail across seeds.
    tail_positions = set()
    for seed in range(6):
        env = apply_noise(base_tools, _cfg(count=1, position="tail",
                                           tool_order="shuffle", seed=seed))
        roles = _roles(env)
        d_idx = roles.index(SOURCE_DISTRACTOR)
        tail_positions.add(d_idx)
    # the distractor lands at more than one distinct index -> not tail-locked
    assert len(tail_positions) > 1


def test_aligned_similarity_mimics_real(base_tools):
    env = apply_noise(base_tools, _cfg(count=2, similarity="aligned"))
    dtools = [t for t in env.tools if env.manifest[t.name].source == SOURCE_DISTRACTOR]
    assert len(dtools) == 2
    real_names = {t.name for t in base_tools}
    for t in dtools:
        meta = env.manifest[t.name].meta
        assert meta["similarity"] == "aligned"
        # aligned distractor declares which real tool it imitates
        assert meta["imitates"] in real_names


def test_similarity_recorded_in_manifest(base_tools):
    env = apply_noise(base_tools, _cfg(count=1, similarity="obvious"))
    dtool = [t for t in env.tools if env.manifest[t.name].source == SOURCE_DISTRACTOR][0]
    assert env.manifest[dtool.name].meta["similarity"] == "obvious"
