//https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/framework/gradients_test.cc
//https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2
//https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2

#include <vector>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor_shape.h>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph_def_util.h>
//#include <tensorflow/core/platform/status.h>

#include <tensorflow/c/c_api.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

// Example:
//      ^             ^
//    dy|           dx|        (MatMul Gradient Graph)
//      |             |
//   MatMul_1      MatMul_2
//   ^   ^          ^    ^
//   |   |----------|    |
//   |        ^          |
//   |      dz|          |
//   |        |          |
//   |     Const_3       |
//   |                   |
//   |        ^          |
//   |       z|          |     (MatMul Forward Graph)
//   |        |          |
//   |      MatMul_0     |
//   |     /        \    |
//   |    ^          ^   |
//   |    |          |   |
//   |---x|         y|---|
//        |          |
//        |          |
//      Const_0   Const_1
//

int main(int argc, char **argv)
{
#if 0
  tensorflow::GraphDef gdef;
  {
    tensorflow::Status _status;

    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    tensorflow::PartialTensorShape x_out;
    tensorflow::gtl::ArraySlice<tensorflow::int64> x_shape({1});
    tensorflow::TensorShapeUtils::MakeShape(x_shape, &x_out);
    tensorflow::ops::Placeholder::Attrs x_attrs;
    x_attrs.Shape(x_out);
    auto x = tensorflow::ops::Placeholder(scope, tensorflow::DT_FLOAT, x_attrs);
    x.node()->set_name("input");

    tensorflow::PartialTensorShape y_out;
    tensorflow::gtl::ArraySlice<tensorflow::int64> y_shape({1});
    tensorflow::TensorShapeUtils::MakeShape(y_shape, &y_out);
    tensorflow::ops::Placeholder::Attrs y_attrs;
    y_attrs.Shape(y_out);
    auto y = tensorflow::ops::Placeholder(scope, tensorflow::DT_FLOAT, y_attrs);
    y.node()->set_name("target");

    //Trivial linear model
    auto y_ = tensorflow::ops::Identity(scope,
                                        tensorflow::ops::DenseBincount(
                                            scope,
                                            x,
                                            tensorflow::ops::Const(scope, {1.0}),
                                            tensorflow::ops::Const(scope, {1.0})));
    assert(tensorflow::Status::OK() == _status);                                       
    //y_.node()->set_name("output");

    //Optimize loss
    auto loss = tensorflow::ops::ReduceMean(scope,
                                            tensorflow::ops::Square(
                                                scope,
                                                tensorflow::ops::Sub(scope, y_, y)),
                                            tensorflow::ops::Const(scope, {1}));

    // tf.train.GradientDescentOptimizer
    // Call AddSymbolicGradients.
    std::vector<tensorflow::Output> train_op;
    _status = tensorflow::AddSymbolicGradients(scope, {loss}, {x, y}, &train_op);
    assert(tensorflow::Status::OK() == _status);
    CHECK_NOTNULL(train_op[0].node());

    //
    _status = scope.ToGraphDef(&gdef);
    assert(tensorflow::Status::OK() == _status);
  }

  tensorflow::WriteBinaryProto(tensorflow::Env::Default(), "graph.pb", gdef);
#endif

  //--------------------------------------------------------------

  // Example of training the model created above.

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
    int _res = stat(checkpoint_prefix, &_buf);
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

#if 0

  TF_Status *status = TF_NewStatus();

  // Import the graph.
  TF_Graph *graph;
  {
    TF_Buffer *graph_def;
    {
      int fd = open("graph.pb", 0);
      assert(fd != -1);

      char data[4096];
      ssize_t nread = read(fd, data, 4096);
      assert(nread != -1 && nread < 4096);

      graph_def = TF_NewBufferFromString(data, nread);
    }
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
    graph = TF_NewGraph();
     TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graph_def);
    assert(TF_OK == TF_GetCode(status));
  }

  // Create the session.
  TF_Session *session;
  {
    TF_SessionOptions *opts = TF_NewSessionOptions();
    session = TF_NewSession(graph, opts, status);
    TF_DeleteSessionOptions(opts);
    assert(TF_OK == TF_GetCode(status));
  }

  TF_Output input;
  TF_Output target;
  TF_Output output;
  TF_GraphOperationByName(graph, "input");
#endif

  return 0;
}
