from src.wireless.channel import ChannelSimulator

def test_channel_simulator():
    cfg = {'wireless': {'block_fading_intensity': 1.0, 'base_snr_db': 10.0, 'per_k': 1.0}}
    sim = ChannelSimulator(cfg, num_clients=5)
    stats = sim.sample_round()
    assert len(stats)==5
    for i in range(5):
        assert 0.0 <= stats[i]['per'] <= 1.0
