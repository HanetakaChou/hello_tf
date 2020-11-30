//https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/framework/gradients_test.cc
//https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2
//https://gist.github.com/asimshankar/5c96acd1280507940bad9083370fe8dc

#include <vector>

#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph_def_util.h>

#include <unistd.h>
#include <sys/stat.h>


int main(int argc, char **argv)
{
  // Example of training the model created above.

  char const *checkpoint_dir = "./checkpoints";
  char const *checkpoint_prefix = "./checkpoints/checkpoint";

  tensorflow::Status _status;

  // Import the graph.
  tensorflow::GraphDef graph_def;
  tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "graph.pb", &graph_def);

  std::cout << tensorflow::SummarizeGraphDef(graph_def) << std::endl;

  // Create the session.
  tensorflow::Session *_session = tensorflow::NewSession(tensorflow::SessionOptions());
  _status = _session->Create(graph_def);
  assert(tensorflow::Status::OK() == _status);

  bool restore;
  {
    struct stat _buf;
    int _res = stat(checkpoint_dir, &_buf);
    restore = (0 == _res);
  }

  if (restore)
  {
    // Restoring model weights from checkpoint
    tensorflow::Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
    t.scalar<tensorflow::tstring>()() = checkpoint_prefix;
    _status = _session->Run(
        {{"save/Const", t}},
        {},
        {"save/restore_all"},
        NULL);
    assert(tensorflow::Status::OK() == _status);
  }
  else
  {
    // Initializing model weights
    _status = _session->Run(
        {},       //inputs
        {},       //output_tensor_names
        {"init"}, //target_node_names
        NULL      //outputs
    );
    assert(tensorflow::Status::OK() == _status);
  }

  //Initial predictions
  {
    std::vector<float> const batch({1.0, 2.0, 3.0});

    tensorflow::Tensor testdata(tensorflow::DT_FLOAT, tensorflow::TensorShape({(int)batch.size(), 1, 1}));
    for (int i = 0; i < batch.size(); ++i)
    {
      testdata.flat<float>()(i) = batch[i];
    }

    std::vector<tensorflow::Tensor> out_tensors;
    _status = _session->Run(
        {{"input", testdata}}, //inputs
        {"output"},            //output_tensor_names
        {},                    //target_node_names
        &out_tensors           //outputs
    );
    assert(tensorflow::Status::OK() == _status);

    for (int i = 0; i < batch.size(); ++i)
    {
      std::cout << "\t x = " << batch[i]
                << ", predicted y = " << out_tensors[0].flat<float>()(i)
                << "\n";
    }
  }

  //Training for a few steps
  for (int i = 0; i < 200; ++i)
  {
    std::vector<float> train_inputs;
    std::vector<float> train_targets;
    for (int j = 0; j < 10; j++)
    {
      train_inputs.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
      train_targets.push_back(3 * train_inputs.back() + 2);
    }

    tensorflow::Tensor input_batch(tensorflow::DT_FLOAT, tensorflow::TensorShape({(int)train_inputs.size(), 1, 1}));
    for (int i = 0; i < train_inputs.size(); ++i)
    {
      input_batch.flat<float>()(i) = train_inputs[i];
    }

    tensorflow::Tensor target_batch(tensorflow::DT_FLOAT, tensorflow::TensorShape({(int)train_targets.size(), 1, 1}));
    for (int i = 0; i < train_targets.size(); ++i)
    {
      target_batch.flat<float>()(i) = train_targets[i];
    }

    _status = _session->Run(
        {{"input", input_batch}, {"target", target_batch}},
        {},
        {"train"},
        NULL);
    assert(tensorflow::Status::OK() == _status);
  }

  // Updated predictions
  {
    std::vector<float> const batch({1.0, 2.0, 3.0});

    tensorflow::Tensor testdata(tensorflow::DT_FLOAT, tensorflow::TensorShape({(int)batch.size(), 1, 1}));
    for (int i = 0; i < batch.size(); ++i)
    {
      testdata.flat<float>()(i) = batch[i];
    }

    std::vector<tensorflow::Tensor> out_tensors;
    _status = _session->Run(
        {{"input", testdata}}, //inputs
        {"output"},            //output_tensor_names
        {},                    //target_node_names
        &out_tensors           //outputs
    );
    assert(tensorflow::Status::OK() == _status);

    for (int i = 0; i < batch.size(); ++i)
    {
      std::cout << "\t x = " << batch[i]
                << ", predicted y = " << out_tensors[0].flat<float>()(i)
                << "\n";
    }
  }

  //Saving checkpoint
  {
    tensorflow::Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
    t.scalar<tensorflow::tstring>()() = checkpoint_prefix;
    _status = _session->Run(
        {{"save/Const", t}},
        {},
        {"save/control_dependency"},
        NULL);
    assert(tensorflow::Status::OK() == _status);
  }

  return 0;
}
