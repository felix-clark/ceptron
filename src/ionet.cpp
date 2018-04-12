#include "ionet.hpp"
#include "log.hpp"
#include <fstream>

#if (not defined(__clang__)) && defined(__GNUC__) && __GNUC__ < 5
#define NO_HEXFLOAT
// we need a workaround with high-precision output
#include <iomanip>
#endif


namespace {
  using namespace ceptron;
}

void ceptron::toFile(const ceptron::ArrayX& net, const std::string& fname) {
  // ios::trunc erases any previous content in the file.
  std::ofstream fout(fname , std::ios::binary | std::ios::trunc );
  if (!fout.is_open()) {
    LOG_ERROR("could not open file " << fname << " for writing.");
    return;
  }
  for (int i=0; i<net.size(); ++i) {
    // we do want to go hexfloat, otherwise we suffer a precision loss
    // removing newline doesn't work even w/ binary, possibly because of the issue discussed in fromFile().

    // hexfloat isn't implemented in g++4, but we could put it to use there.
#ifdef NO_HEXFLOAT
    // for every 4 bytes of a floating point representation, there are typically 9 decimal digits of precision at a maximum
    fout << std::setprecision(sizeof(ceptron::scalar)*9/4) << net(i) << '\n';
#else
    fout << std::hexfloat << net(i) << '\n';
#endif
  }
  fout.close();
} // toFile()
  
ceptron::ArrayX ceptron::fromFile(const std::string& fname) {
  // some metadata at the top might be nice for verification, but now that we're just using raw arrays perhaps it's less necessary
  std::ifstream fin(fname, std::ios::binary);
  if (!fin.is_open()) {
    LOG_ERROR("could not open file " << fname << " for reading.");
    throw std::runtime_error("failed to open file");
  }
  std::string line;
  std::vector<ceptron::scalar> vals;
  while (std::getline(fin, line)) {
    // the fact that this one line doesn't work is actually either a bug in g++ or a flaw in the standards.
    // in >> std::hexfloat >> data(i);
    // it may be fixed in C++ 17 but g++ isn't there yet.
    // see: https://stackoverflow.com/questions/42604596/read-and-write-using-stdhexfloat
    
    std::stringstream s;
#ifdef NO_HEXFLOAT
    s << line;
#else
    s << std::hexfloat << line;
#endif
    vals.push_back(std::strtold(s.str().data(), nullptr));
  }
    
  fin.close();
    
  ArrayX data = Eigen::Map<ArrayX>(vals.data(), vals.size());;
  return data;
} // fromFile()

