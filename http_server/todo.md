 - union find for game state management
 - merging mcts nodes to save memory and improve performance (making a DAG esentially)
 - prallel mcts and virtual loss


 ONNX / TensorRT: For maximum inference speed on NVIDIA GPUs, exporting the model to ONNX format and then optimizing it with TensorRT often yields the best results.



# 7x7:

lookup table parameter
agent NUM_POSSIBLE_SCORES = int(30.5 * 2 + 1)