"""Shared muscle definitions and literature-informed articulatory priors.

MUSCLE ORDER IS FIXED and must match the ArtiSynth HexTongueDemo excitation
order used everywhere else in the repo (configs/artisynth/default.yaml).
"""

MUSCLE_NAMES = [
    "GGP",    # genioglossus posterior  -> tongue body advance / dorsum raise
    "GGM",    # genioglossus medial
    "GGA",    # genioglossus anterior   -> tongue tip/blade lower, body forward
    "STY",    # styloglossus            -> tongue body up+back (posterior elevation)
    "GH",     # geniohyoid              -> hyoid forward/up (floor of mouth)
    "MH",     # mylohyoid               -> floor of mouth elevation, tongue up
    "HG",     # hyoglossus              -> tongue body down + root retract
    "VERT",   # verticalis (intrinsic)  -> flattens tongue, deepens groove
    "TRANS",  # transversus (intrinsic) -> narrows + lengthens/protrudes tongue
    "IL",     # inferior longitudinal   -> tip down / retract
    "SL",     # superior longitudinal   -> tip up / dorsum concave
]
N_MUSCLES = len(MUSCLE_NAMES)
MI = {m: i for i, m in enumerate(MUSCLE_NAMES)}

# ---------------------------------------------------------------------------
# Literature-informed articulatory synergy anchors.
#
# IMPORTANT / HONESTY NOTE:
#   These are APPROXIMATE qualitative priors distilled from EMG and biomechanical
#   modelling literature. They are NOT measured ground truth for this particular
#   ArtiSynth model. Their ONLY role here is to bias sampling DENSITY toward the
#   region of activation space that plausibly matters for speech / oral behaviour.
#   They are never used as supervision, and never as an evaluation target.
#   Treat this table as a tunable config. Edit freely.
#
# REFERENCES (근거 문헌):
#   Vowels
#     - Buchaillard, Perrier & Payan 2009, JASA 126(4):2033 — "A biomechanical
#       model of cardinal vowel production": GGP for high-front /i/, STY for
#       high-back /u,o/, HG for low /a/. (같은 계열 3D FEM 혀 모델)
#     - Takano & Honda 2007, "An MRI analysis of the extrinsic tongue muscles
#       during vowel production" (Speech Communication).
#     - Baer, Alfonso & Honda 1988, JASA — vowel EMG of GG(a/p), HG, STY.
#   Consonants
#     - Harandi et al. 2017, JASA 141(4) — "Variability in muscle activation of
#       simple speech motions": GGa 는 설첨을 구개로 올릴 때 항상 활성; /s/ 에서
#       VERT+TRANS 로 groove 형성.
#     - "The Compartmental Tongue" 2024, JSLHR — 내재근 구획별 기능(SL 설첨 상승,
#       IL 설첨 하강/후퇴, VERT groove, TRANS 협착/신장).
#     - Zhou et al. 2007, Interspeech — "retroflex" vs "bunched" American /r/
#       (설첨 말아올림 = SL+IL vs 설체 융기 = GGP+IL).
#   Non-speech / physiological
#     - Cunningham & Basmajian 1969 — GG/GH EMG during deglutition.
#     - Orsbon et al. 2020, Sci. Rep. — XROMM: 삼킴 시 설근 후퇴(HG+STY) 수압 기전.
#     - Feeding EMG (intrinsic+extrinsic) — Liu et al. 2007 (pig model).
#   Intrinsic muscle actions: Takemoto 2001; Kier & Smith 1985 (muscular hydrostat).
#
# SEPARABILITY NOTE (중심 겹침 정리):
#   38개 anchor 중, 방향공간(L1-normalized)에서 정준 anchor 와 거의 겹치던 6개를
#   제거해 32개로 정리했다. 제거된 것은 혀 근육 11개만으로는 정준음과 구별 불가한
#   소리들(비음성/성대/원순 등 비-혀 자질만 다름)이다:
#     vowel_ih(/ɪ/)→vowel_i,  vowel_oo(/ʊ/)→vowel_u,  vowel_er(/ɝ/)→cons_r_bunched,
#     cons_ng(/ŋ/)→cons_k_g,  cons_y(/j/)→vowel_i,     swallow_base_retract→retraction.
#   결과: 최소 중심간 거리 0.022→0.119, ANCHOR 샘플의 최근접-중심 구별도 59%→68%
#   (교란을 tight 로 좁히면 95%). 32개 중심은 서로 뚜렷이 분리된다.
# ---------------------------------------------------------------------------
ARTICULATORY_ANCHORS = {
    # --- cardinal vowels ---
    "vowel_i": {"GGP": 0.55, "GGM": 0.25, "SL": 0.15, "TRANS": 0.10},          # high front
    "vowel_a": {"HG": 0.50, "GGA": 0.15, "STY": 0.10},                          # low back
    "vowel_u": {"STY": 0.45, "GGP": 0.30, "TRANS": 0.20, "MH": 0.10},           # high back rounded
    "vowel_e": {"GGP": 0.35, "GGM": 0.20, "SL": 0.10},
    "vowel_o": {"STY": 0.35, "HG": 0.20, "GGP": 0.15},
    "vowel_schwa": {"GGP": 0.12, "GGM": 0.08},                                  # near-neutral
    # --- additional English monophthongs (lax + central + rhotic) ---
    #   lax 모음은 대응 tense 모음의 약화판(조음 방향 유지, 진폭 하향). 근거: 위 REFERENCES.
    "vowel_eh": {"GGM": 0.22, "GGA": 0.15, "GGP": 0.12},                        # /ɛ/ bet  (mid-low front)
    "vowel_ae": {"GGA": 0.30, "HG": 0.25, "MH": 0.10},                          # /æ/ bat  (low front)
    "vowel_uh": {"HG": 0.20, "GGM": 0.12, "GGA": 0.08},                         # /ʌ/ but  (mid-central)
    "vowel_aw": {"STY": 0.30, "HG": 0.30, "GGP": 0.10},                         # /ɔ/ thought (low-mid back rounded)
    "vowel_ic": {"GGP": 0.30, "STY": 0.20, "GGM": 0.12},                        # /ɨ/ central-high (roses)
    # 제거됨(중심 겹침): vowel_ih/ɪ→vowel_i, vowel_oo/ʊ→vowel_u, vowel_er/ɝ→cons_r_bunched
    #                  (혀 근육만으로는 tense 모음과 구별 불가 → 중복 중심 제거)
    # --- consonant constrictions ---
    "cons_t_d": {"SL": 0.50, "GGA": 0.25, "TRANS": 0.20, "MH": 0.15},           # alveolar tip stop
    "cons_s_z": {"SL": 0.40, "GGA": 0.30, "VERT": 0.30, "TRANS": 0.25},         # grooved sibilant
    "cons_k_g": {"STY": 0.50, "GGP": 0.35, "MH": 0.10},                         # velar dorsum
    "cons_l": {"SL": 0.45, "IL": 0.20, "GGA": 0.20},                            # lateral
    "cons_r_bunched": {"GGP": 0.40, "IL": 0.30, "TRANS": 0.25, "HG": 0.15},
    "cons_sh": {"SL": 0.30, "GGP": 0.30, "TRANS": 0.20},
    "cons_n": {"SL": 0.40, "MH": 0.25, "GGA": 0.20},
    # --- additional English consonants ---
    "cons_th": {"SL": 0.40, "GGA": 0.30, "IL": 0.12},                           # /θ,ð/ dental (설첨 치아)
    "cons_ch_dg": {"SL": 0.45, "GGP": 0.25, "TRANS": 0.25, "GGA": 0.15},        # /tʃ,dʒ/ 파찰 (stop+sh 혼합)
    "cons_w": {"STY": 0.40, "GGP": 0.30, "TRANS": 0.15},                        # /w/ 연구개 활음 (설체 후상방)
    "cons_r_retroflex": {"SL": 0.35, "IL": 0.35, "GGP": 0.25},                  # retroflex /r/ (설첨 말아올림)
    # 제거됨(중심 겹침): cons_ng/ŋ→cons_k_g (연구개 자세 동일), cons_y/j→vowel_i (고전설 동일)
    # --- non-speech / boundary postures ---
    "protrusion": {"TRANS": 0.60, "GGA": 0.30},
    "retraction": {"HG": 0.40, "IL": 0.35, "STY": 0.25},
    "tip_elevate": {"SL": 0.60},
    "flatten": {"VERT": 0.55},
    "swallow_like": {"MH": 0.45, "GH": 0.40, "HG": 0.25, "SL": 0.20},
    # --- additional physiological / feeding gestures ---
    "swallow_bolus_push": {"MH": 0.40, "GH": 0.35, "GGA": 0.30, "SL": 0.25},    # 구강기: 설배로 구개 압박(전→후)
    "groove_deep": {"VERT": 0.55, "TRANS": 0.35},                               # 깊은 세로 groove (설측 하강)
    # 제거됨(중심 겹침): swallow_base_retract → retraction/vowel_aw (설근 후퇴 방향 동일)
    "tip_curl_back": {"IL": 0.45, "SL": 0.30, "GGP": 0.15},                     # 설첨 말아 뒤로(핥기/retroflex 자세)
    "palate_brace": {"MH": 0.40, "GGA": 0.30, "TRANS": 0.20, "GGM": 0.15},      # 저작 중 설체 고정(bracing)
    "suction_click": {"STY": 0.45, "MH": 0.35, "GGP": 0.20},                    # 흡인/설배 흡착(velaric)
}


def anchor_to_vec(d, cap=1.0):
    import numpy as np
    v = np.zeros(N_MUSCLES)
    for m, a in d.items():
        v[MI[m]] = a
    return np.clip(v, 0.0, cap)
