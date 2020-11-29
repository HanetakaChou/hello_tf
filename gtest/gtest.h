#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_H_

#include <assert.h>

#if !GTEST_DONT_DEFINE_ASSERT_EQ
#define ASSERT_EQ(val1, val2) assert((val1) == (val2))
#endif

#define EXPECT_TRUE(condition) assert((condition)), std::cout

#endif