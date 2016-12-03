
#include <gtest/gtest.h>
#include <litiv/utils.hpp>

TEST(distances, l1dist_zero) {
	EXPECT_EQ(lv::L1dist(5.0f,5.0f),0.0f);
}