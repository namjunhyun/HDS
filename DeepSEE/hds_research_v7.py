# ── 모델 아키텍처 재구성 (checkpoint 기준으로 강제 맞춤) ──

feat_dim = windows.shape[2]

# fc1 shape → ca_d_model 계산
fc1_w = state_dict['ca_regressor.fc1.weight']
fc1_out, fc1_in = fc1_w.shape
ca_d_model = fc1_in // 34

print(f"  fc1 shape     : {fc1_w.shape}")
print(f"  ca_d_model    : {ca_d_model}")

# ✅ 🔥 핵심: checkpoint 구조 그대로 복원
pd_config = TimesformerConfig(
    image_size=128,
    patch_size=8,
    num_channels=3,
    num_frames=4,
    num_hidden_layers=3,
    hidden_size=192,
    intermediate_size=256,
)

ts_config = PatchTSMixerConfig(
    context_length=30,
    patch_len=30,            # 🔥 무조건 30
    num_input_channels=feat_dim,
    d_model=64,              # 🔥 무조건 64
)

ca_config = MultiModalCrossAttentionConfig(
    ca_d_model=ca_d_model,
    reg_d_fc=fc1_out,
    ts_num_input_channels=feat_dim,
    ts_d_model=64,           # 🔥 반드시 64
    pd_width=96,
    pd_height=128,
    pd_d_model=192,
    ts_context_length=30,
)

ca_config.pe_max_len = 10000

model = DeepSEEModel(pd_config, ts_config, ca_config).to(device)

# ✅ 로드
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(f"  Missing keys  : {len(missing)}")
print(f"  Unexpected    : {len(unexpected)}")
