/*
 * Copyright 1996-2020 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * (c) 2006 - 2008 EPFL, Lausanne, Switzerland
 * Thomas Lochmatter
 * Adapted by Nicolas Heiniger for the e-puck
 */

#include "odometry.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "errno.h"

#define PI 3.14159265358979

// Original global calibration parameters (keep your values)
float increments_per_tour  = 976.0f;
float axis_wheel_ratio     = 1.293186f;
float wheel_diameter_left  = 0.04105f;
float wheel_diameter_right = 0.04105f;
float scaling_factor       = 0.974f;

// ===== Runtime setters & recomputation =====
void odometry_set_wheel_diameters(float left, float right) {
  wheel_diameter_left  = left;
  wheel_diameter_right = right;
}

void odometry_recompute_config(struct sOdometryTrack *ot) {
  // Recompute odometry conversion factors from current calibration values
  ot->configuration.wheel_distance =
      axis_wheel_ratio * scaling_factor * (wheel_diameter_left + wheel_diameter_right) / 2.0f;
  ot->configuration.wheel_conversion_left  =
      wheel_diameter_left  * scaling_factor * PI / increments_per_tour;
  ot->configuration.wheel_conversion_right =
      wheel_diameter_right * scaling_factor * PI / increments_per_tour;
}

void odometry_track_init() {
}

int odometry_track_start_pos(struct sOdometryTrack *ot, int pos_left, int pos_right) {
  ot->result.x = 0.0f;
  ot->result.y = 0.0f;
  ot->result.theta = 0.0f;

  ot->state.pos_left_prev = pos_left;
  ot->state.pos_right_prev = pos_right;

  // Initial odometry configuration from calibration parameters
  ot->configuration.wheel_distance =
      axis_wheel_ratio * scaling_factor * (wheel_diameter_left + wheel_diameter_right) / 2.0f;
  ot->configuration.wheel_conversion_left  =
      wheel_diameter_left  * scaling_factor * PI / increments_per_tour;
  ot->configuration.wheel_conversion_right =
      wheel_diameter_right * scaling_factor * PI / increments_per_tour;

  return 1;
}

void odometry_track_step_pos(struct sOdometryTrack *ot, int pos_left, int pos_right) {
  int delta_pos_left, delta_pos_right;
  float delta_left, delta_right, delta_theta, theta2;
  float delta_x, delta_y;

  delta_pos_left  = pos_left  - ot->state.pos_left_prev;
  delta_pos_right = pos_right - ot->state.pos_right_prev;

  delta_left  = delta_pos_left  * ot->configuration.wheel_conversion_left;
  delta_right = delta_pos_right * ot->configuration.wheel_conversion_right;

  delta_theta = (delta_right - delta_left) / ot->configuration.wheel_distance;
  theta2 = ot->result.theta + delta_theta * 0.5f;

  delta_x = (delta_left + delta_right) * 0.5f * cosf(theta2);
  delta_y = (delta_left + delta_right) * 0.5f * sinf(theta2);

  ot->result.x     += delta_x;
  ot->result.y     += delta_y;
  ot->result.theta += delta_theta;

  if (ot->result.theta > PI)
    ot->result.theta -= 2.0f * PI;
  if (ot->result.theta < -PI)
    ot->result.theta += 2.0f * PI;

  ot->state.pos_left_prev  = pos_left;
  ot->state.pos_right_prev = pos_right;
}
