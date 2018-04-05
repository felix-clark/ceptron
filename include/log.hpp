#pragma once

#ifdef BOOST_AVAILABLE
// use the boost logging
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
// #include <boost/log/sources/basic_logger.hpp> // could be useful in class defs in the future

#define LOG_TRACE(ARG) do {BOOST_LOG_TRIVIAL(trace) << ARG;} while (false)
#define LOG_DEBUG(ARG) do {BOOST_LOG_TRIVIAL(debug) << ARG;} while (false)
#define LOG_INFO(ARG) do {BOOST_LOG_TRIVIAL(info) << ARG;} while (false)
#define LOG_WARNING(ARG) do {BOOST_LOG_TRIVIAL(warning) << ARG;} while (false)
#define LOG_ERROR(ARG) do {BOOST_LOG_TRIVIAL(error) << ARG;} while (false)
#define LOG_FATAL(ARG) do {BOOST_LOG_TRIVIAL(fatal) << ARG;} while (false)

#define SET_LOG_LEVEL(LVL) do {boost::log::core::get()->set_filter \
      ( boost::log::trivial::severity >= boost::log::trivial:: LVL );} while (false)

#else
#include <iostream>
// without boost we'll use our own simplified iostream version
// it will just use a global logging level constant.

namespace logging {
  enum level : int {fatal, error, warning, info, debug, trace};
  static level global_log_level = info;
}
#define LOG_TRACE(ARG) do {if (logging::global_log_level >= logging::trace)\
      std::cout << "[trace]   " << ARG << std::endl;} while (false)
#define LOG_DEBUG(ARG) do {if (logging::global_log_level >= logging::debug)\
      std::cout << "[debug]   " << ARG << std::endl;} while (false)
#define LOG_INFO(ARG) do {if (logging::global_log_level >= logging::info)\
      std::cout << "[info]    " << ARG << std::endl;} while (false)
#define LOG_WARNING(ARG) do {if (logging::global_log_level >= logging::warning)\
      std::cout << "[warning] " << ARG << std::endl;} while (false)
#define LOG_ERROR(ARG) do {if (logging::global_log_level >= logging::error)\
      std::cout << "[error]   " << ARG << std::endl;} while (false)
#define LOG_FATAL(ARG) do {if (logging::global_log_level >= logging::fatal)\
      std::cout << "[fatal]   " << ARG << std::endl;} while (false)

#define SET_LOG_LEVEL(LVL) do {logging::global_log_level = logging::LVL;} while (false)

#endif
