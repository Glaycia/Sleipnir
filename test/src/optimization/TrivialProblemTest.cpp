// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CmdlineArguments.hpp"

TEST(TrivialProblemTest, Empty) {
  sleipnir::OptimizationProblem problem;

  auto status =
      problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);
}

TEST(TrivialProblemTest, NoCostUnconstrained) {
  {
    sleipnir::OptimizationProblem problem;

    auto X = problem.DecisionVariable(2, 3);

    auto status =
        problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.costFunctionType);
    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

    for (int row = 0; row < X.Rows(); ++row) {
      for (int col = 0; col < X.Cols(); ++col) {
        EXPECT_EQ(0.0, X.Value(row, col));
      }
    }
  }

  {
    sleipnir::OptimizationProblem problem;

    auto X = problem.DecisionVariable(2, 3);
    X.SetValue(Eigen::Matrix<double, 2, 3>{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}});

    auto status =
        problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.costFunctionType);
    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

    for (int row = 0; row < X.Rows(); ++row) {
      for (int col = 0; col < X.Cols(); ++col) {
        EXPECT_EQ(1.0, X.Value(row, col));
      }
    }
  }
}
