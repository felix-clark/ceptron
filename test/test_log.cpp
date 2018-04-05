#include "log.hpp"


int main() {

  SET_LOG_LEVEL(debug);
  LOG_TRACE( "trace loggin'" );
  LOG_DEBUG( "debug loggin'" );
  LOG_INFO( "info loggin'" );
  LOG_WARNING( "warning loggin'" );
  LOG_ERROR( "error loggin'" );
  LOG_FATAL( "fatal loggin'" );
  return 0;
}
