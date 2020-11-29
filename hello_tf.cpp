//https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/framework/gradients_test.cc

#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/lib/core/status_test_util.h>
using string = std::string;
#include <tensorflow/core/util/equal_graph_def.h>

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
  tensorflow::Scope scope_expected_ = tensorflow::Scope::NewRootScope();
  {
    auto x = tensorflow::ops::Const(scope_expected_, {{1.0, 2.0}, {3.0, 4.0}});
    auto y = tensorflow::ops::Const(scope_expected_, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = tensorflow::ops::MatMul(scope_expected_, x, y);
    TF_ASSERT_OK(scope_expected_.status());
    CHECK_NOTNULL(z.node());

    // Construct backward graph.
    // The gradients function adds a OnesLike to create a dz of ones with the
    // shape of z.
    auto dz = tensorflow::ops::OnesLike(scope_expected_, z);
    auto dx = tensorflow::ops::MatMul(scope_expected_, dz, y, tensorflow::ops::MatMul::TransposeB(true));
    auto dy = tensorflow::ops::MatMul(scope_expected_, x, dz, tensorflow::ops::MatMul::TransposeA(true));
  }

  tensorflow::Scope scope_test_ = tensorflow::Scope::NewRootScope();
  {
    auto x = tensorflow::ops::Const(scope_test_, {{1.0, 2.0}, {3.0, 4.0}});
    auto y = tensorflow::ops::Const(scope_test_, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = tensorflow::ops::MatMul(scope_test_, x, y);
    TF_ASSERT_OK(scope_test_.status());
    CHECK_NOTNULL(z.node());

    // Call AddSymbolicGradients.
    std::vector<tensorflow::Output> grad_outputs;
    // tf.train.GradientDescentOptimizer
    TF_ASSERT_OK(tensorflow::AddSymbolicGradients(scope_test_, {z}, {x, y}, &grad_outputs));
  }

  tensorflow::GraphDef gdef_test;
  TF_ASSERT_OK(scope_test_.ToGraphDef(&gdef_test));
  tensorflow::GraphDef gdef_exp;
  TF_ASSERT_OK(scope_expected_.ToGraphDef(&gdef_exp));

  TF_EXPECT_GRAPH_EQ(gdef_exp, gdef_test);

  tensorflow::WriteBinaryProto(tensorflow::Env::Default(), "graph.pb", gdef_test);

  return 0;
}
