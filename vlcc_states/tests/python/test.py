import torch
import vlcc_states

def test_multiple_tensors():
    # Create multiple tensors
    tensor1 = torch.randn((300), pin_memory=True, device='cpu')
    tensor2 = torch.randn((1<<20), device='cuda')
    mystr = "This is a test string"*10

    sp1 = vlcc_states.ObjectProvider("tensor1", vlcc_states.TIER_TYPES.HOST_PINNED_TIER)
    sp1.register_state(tensor1)

    sp2 = vlcc_states.ObjectProvider("tensor2", vlcc_states.TIER_TYPES.GPU_TIER)
    sp2.register_state(tensor2)

    sp3 = vlcc_states.ObjectProvider("string", vlcc_states.TIER_TYPES.HOST_UNPINNED_TIER)
    sp3.register_state(mystr)

    sp1_chunk = sp1.get_next_chunk(sp1.get_tier())
    print("SP1 chunk size: ", sp1_chunk.size())

    # Create a composite state provider
    composite_provider = vlcc_states.CompositeProvider("composite", vlcc_states.TIER_TYPES.COMPOSITE_TIER)
    composite_provider.register_provider(sp1)
    composite_provider.register_provider(sp2)
    composite_provider.register_provider(sp3)

    print("Can register multiple tensors and strings in a composite state provider.")

if __name__ == "__main__":
    test_multiple_tensors()
    print("Test completed successfully.")