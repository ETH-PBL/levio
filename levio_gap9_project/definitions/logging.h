// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __LOGGING_H__
#define __LOGGING_H__

#include "pmsis.h"

/*
 * Logging level definitions and macros for conditional logging.
 *
 * ERROR_LEVEL, WARNING_LEVEL, INFO_LEVEL, DEBUG_LEVEL:
 *   - Integer constants representing different logging levels.
 * LOG_LEVEL:
 *   - Sets the current logging level. Only messages at or above this level will be logged.
 *   - Configured in config.h.
 * LOG_TIMING_ENABLED:
 *   - Enables or disables timing logs.
 *   - Configured in config.h.
 * LOG_TIMING(...):
 *   - Logs timing information if LOG_TIMING_ENABLED is set.
 * LOG_ERROR(...):
 *   - Logs error messages if LOG_LEVEL is set to ERROR_LEVEL or higher.
 * LOG_WARNING(...):
 *   - Logs warning messages if LOG_LEVEL is set to WARNING_LEVEL or higher.
 * LOG_INFO(...):
 *   - Logs informational messages if LOG_LEVEL is set to INFO_LEVEL or higher.
 * LOG_DEBUG(...):
 *   - Logs debug messages if LOG_LEVEL is set to DEBUG_LEVEL or higher.
 */

#define ERROR_LEVEL     0x00
#define WARNING_LEVEL   0x01
#define INFO_LEVEL      0x02
#define DEBUG_LEVEL     0x03

/* LOG_LEVEL and LOG_TIMING_ENABLED are defined in config.h */
#include "config.h"

#define LOG_TIMING(...)  do {if (LOG_TIMING_ENABLED) printf(__VA_ARGS__);} while(0)
#define LOG_ERROR(...)   do {if (LOG_LEVEL >= ERROR_LEVEL) printf(__VA_ARGS__);} while(0)
#define LOG_WARNING(...) do {if (LOG_LEVEL >= WARNING_LEVEL) printf(__VA_ARGS__);} while(0)
#define LOG_INFO(...)    do {if (LOG_LEVEL >= INFO_LEVEL) printf(__VA_ARGS__);} while(0)
#define LOG_DEBUG(...)   do {if (LOG_LEVEL >= DEBUG_LEVEL) printf(__VA_ARGS__);} while(0)

#endif /* __LOGGING_H__ */