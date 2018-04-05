#include "global.hpp"
#include "training.hpp"
#include "train_dyn.hpp"
#include "slfn.hpp"
#include "ffn_dyn.hpp"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip> // for setprecision
#include <cstdlib> // for srand
//#include <ctime> // for time

namespace {
  using namespace ceptron;
  
  // constants for this source only
  constexpr size_t Nin = 16; // there are 18-1 features; 16 bools and 1 int
  constexpr size_t Nout = 7; // there are 7 possible types of animals in this set
  constexpr size_t Nh = 8; // number of nodes in hidden layer
  constexpr RegressionType Reg=RegressionType::Categorical;
  // constexpr InternalActivator Act=InternalActivator::Softplus;
  constexpr InternalActivator Act=InternalActivator::Tanh;
  using Net = SlfnStatic<Nin, Nout, Nh>;

  namespace logging = boost::log;
  using std::string;
  using std::pair;
  using std::vector;
  
} // namespace

// get batch that is full data in test sample
pair< BatchVec<Nin>, BatchVec<Nout> > readFromFile(string fname="../data/zoo.data");

// returns a string in english of the class type
string toString(int class_type);

// we'll pre-define a few examples that aren't in the training set
Vec<Nin> x_dog();
Vec<Nin> x_woodpecker();
Vec<Nin> x_salamander();

void printPredictions(const FfnDyn& net, const ArrayX& pars) {
  vector<pair<string, VecX>> animals = {
    {"dog", x_dog()},
    {"woodpecker", x_woodpecker()},
    {"salamander", x_salamander()},
  };
  for (const auto& animal : animals) {
    BOOST_LOG_TRIVIAL(info) << animal.first << ":";
    ArrayX prediction = net.prediction(pars, animal.second).array();
    auto candidateIndices = (prediction > 0.2);
    for (size_t i=0; i<Nout; ++i) {
      if (candidateIndices[i]) {
	BOOST_LOG_TRIVIAL(info) << "  " << toString(i+1) << "  " << prediction[i]*100 << "%";
      }
    }
    // a full printout
    BOOST_LOG_TRIVIAL(trace) << animal.first << ": " << prediction.transpose();
  }
}

int main(int argc, char** argv) {
  logging::core::get()->set_filter
    (logging::trivial::severity >= logging::trivial::debug);
  
  if (argc <= 1) {
    BOOST_LOG_TRIVIAL(info) << "usage: " << argv[0] << " <path>/<to>/zoo.data";
    return 1;
  }
  string datafile = argv[1];

  auto datapair = readFromFile(datafile);
  BatchVec<Nin> xb = datapair.first;
  BatchVec<Nout> yb = datapair.second;

  // seed random initializations. Eigen uses std::rand(), which is seeded via srand().
  // std::srand( time(nullptr) );
  std::srand( 3490 );
  double l2reg = 0.1;
  int numEpochs = 320; // AdaDelta trains pretty quickly, and probably starts to overfit. it does do better at categorizing a salamander as an amphibian rather than a reptile if we let it run more.
  
  FfnDyn netd(Reg, Act, Nin, Nh, Nout);
  netd.setL2Reg(l2reg);
  auto parsd = netd.randomWeights();
  // BOOST_LOG_TRIVIAL(trace) << parsd; // looks fine
  AdaDelta msd(netd.num_weights());
  
  BOOST_LOG_TRIVIAL(info) << std::setprecision(4); // set this once
  BOOST_LOG_TRIVIAL(info) << "running test on dynamic version";
  BOOST_LOG_TRIVIAL(debug) << "parameter space has dimension " << netd.num_weights();

  BOOST_LOG_TRIVIAL(debug) << "pre-training predictions (should just be random):";
  printPredictions(netd, parsd);
  
  for (int i_ep=0; i_ep<numEpochs; ++i_ep) {
    if (i_ep % (numEpochs/4) == numEpochs/8) {
      BOOST_LOG_TRIVIAL(info) << "beginning " << i_ep << "th epoch";
      BOOST_LOG_TRIVIAL(info) << "cost function: " << netd.costFunc(parsd, xb, yb);
      BOOST_LOG_TRIVIAL(debug) << "mid-training predictions:";
      printPredictions(netd, parsd);
    }
    trainFfnDyn(netd, parsd, msd, xb, yb );
  }
  BOOST_LOG_TRIVIAL(info) << "post-training predictions:";
  printPredictions(netd, parsd);
  
  
#ifndef NOSTATIC
  
  // BOOST_LOG_TRIVIAL( debug ) << '\n' << xb.transpose();
  // BOOST_LOG_TRIVIAL( debug ) << '\n' << yb.transpose();

  // since this dataset is rather small, we'll do full batch gradient descent directly.

  Net net;
  net.randomInit(); // should be done by default now, but just to be explicit we'll re-do it here.
  // from a programming perspective it seems preferable to not initialize to random variables, but we have to make sure to remember to randomize every time.
  BOOST_LOG_TRIVIAL(info) << "";
  BOOST_LOG_TRIVIAL(info) << "running test on static version";
  BOOST_LOG_TRIVIAL(info) << "parameter space has dimension " << Net::size;
  // GradientDescent minstep(Net::size); // this could be a choice of a different minimizer
  // AcceleratedGradient minstep(Net::size);
  AdaDelta minstep(Net::size); // this is a decent first choice since it is not supposed to depend strongly on hyperparameters

  BOOST_LOG_TRIVIAL(debug) << "pre-training predictions (should just be random):";
  BOOST_LOG_TRIVIAL(debug) << "dog: " << prediction<Net, Reg, Act>(net, x_dog()).transpose();
  BOOST_LOG_TRIVIAL(debug) << "woodpecker: " << prediction<Net, Reg, Act>(net, x_woodpecker()).transpose();
  BOOST_LOG_TRIVIAL(debug) << "salamander: " << prediction<Net, Reg, Act>(net, x_salamander()).transpose();
  
  for (int i_ep=0; i_ep<numEpochs; ++i_ep) {
    if (i_ep % (numEpochs/4) == numEpochs/8) {
      BOOST_LOG_TRIVIAL(info) << "beginning " << i_ep << "th epoch";
      BOOST_LOG_TRIVIAL(info) << "cost function: " << costFunc<Net, Reg, Act>(net, xb, yb, l2reg);
      BOOST_LOG_TRIVIAL(debug) << "mid-training predictions:";
      BOOST_LOG_TRIVIAL(debug) << "dog: " << prediction<Net, Reg, Act>(net, x_dog()).transpose();
      BOOST_LOG_TRIVIAL(debug) << "woodpecker: " << prediction<Net, Reg, Act>(net, x_woodpecker()).transpose();
      BOOST_LOG_TRIVIAL(debug) << "salamander: " << prediction<Net, Reg, Act>(net, x_salamander()).transpose();
    }
    trainSlfnStatic<Net, Reg, Act>( net, minstep, xb, yb, l2reg );
  }

  BOOST_LOG_TRIVIAL(info) << "post-training predictions:";
  BOOST_LOG_TRIVIAL(info) << "dog: " << prediction<Net, Reg, Act>(net, x_dog()).transpose();
  BOOST_LOG_TRIVIAL(info) << "woodpecker: " << prediction<Net, Reg, Act>(net, x_woodpecker()).transpose();
  BOOST_LOG_TRIVIAL(info) << "salamander: " << prediction<Net, Reg, Act>(net, x_salamander()).transpose();

#else
  BOOST_LOG_TRIVIAL(info) << "Skipped testing static nets.";
#endif // NOSTATIC
  
  return 0;
}


// get batch that is full data in test sample
pair< BatchVec<Nin>, BatchVec<Nout> > readFromFile(string fname) {
  ifstream fin(fname);
  if (!fin.is_open()) {
    BOOST_LOG_TRIVIAL(error) << "Could not open " << fname;
    return std::make_pair( Vec<Nin>::Zero(Nin), Vec<Nout>::Zero(Nout) );
  }
  string line;
  int nlines = 0;
  while (std::getline(fin, line).good()) {
    nlines++;
  }
  BOOST_LOG_TRIVIAL( debug ) << nlines << " lines in file";
  fin.close();
  fin.open(fname);
  Mat<Nin, Eigen::Dynamic> mxs = Mat<Nin, Eigen::Dynamic>::Zero(Nin, nlines);
  Mat<Nout, Eigen::Dynamic> mys = Mat<Nout, Eigen::Dynamic>::Zero(Nout, nlines);
  for (int i_col=0; std::getline(fin, line).good(); ++i_col ) {
    std::istringstream s(line);
    string elem;
    std::getline(s, elem, ','); // this is a throwaway datapoint (animal name); could be printed for debugging
    BOOST_LOG_TRIVIAL(trace) << "reading " << elem;
    std::vector<int> ins;
    while ( std::getline(s, elem, ',').good() ) {
      // this will not read the last element, which is followed by a '\n'
      ins.push_back( std::stoi(elem) );
    }
    for (size_t i=0; i<Nin; ++i) {
      mxs(i,i_col) = ins.at(i);
    }
    std::getline(s, elem);
    int animal_class = std::stoi(elem); // the last element is followed by a newline and represents the animal class
    // BOOST_LOG_TRIVIAL(debug) << "ac minus 1: " << (animal_class-1);
    // BOOST_LOG_TRIVIAL(debug) << Mat<Nout, Nout>::Identity(Nout, Nout)/*.col(animal_class-1)*/;
    // Vec<Nout> class_vec = Mat<Nout, Nout>::Identity(Nout, Nout).col(animal_class-1);
    // BOOST_LOG_TRIVIAL(debug) << "ac: " << animal_class;
    // BOOST_LOG_TRIVIAL(debug) << "cv.T: " << class_vec.transpose();
    // BOOST_LOG_TRIVIAL(debug) << "\n";
    // mys.col(i_col) = class_vec;
    
    // for some reason, trying to grab the i^th column of the identity matrix is not working properly
    mys(animal_class-1, i_col) = 1; // and the rest should be zeros

  }

  return std::make_pair( mxs, mys );
}

string toString(int class_type) {
  switch(class_type) {
  case 1: return "mammal";
  case 2: return "bird";
  case 3: return "reptile";
  case 4: return "fish";
  case 5: return "amphibian";
  case 6: return "bug";
  case 7: return "invertebrate";
  }
  return "unknown";
}

Vec<Nin> x_salamander() {
  Vec<Nin> x;
  x << 0 // hair
    , 0 // feathers
    , 1 // eggs
    , 0 // milk
    , 0 // airborne
    , 1 // aquatic
    , 1 // predator
    , 1 // toothed
    , 1 // backbone
    , 0 // breathes // many salamanders do have lungs, however
    , 0 // venomous // not venomous, but poisonous
    , 0 // fins
    , 4 // legs
    , 1 // tail
    , 0 // domestic
    , 0 // catsize (seems to mean "at least as large as a cat")
    ;
  return x;
}

Vec<Nin> x_dog() {
  Vec<Nin> x;
  x << 1 // hair
    , 0 // feathers
    , 0 // eggs
    , 1 // milk
    , 0 // airborne
    , 0 // aquatic
    , 1 // predator
    , 1 // toothed
    , 1 // backbone
    , 1 // breathes // many salamanders do have lungs, however
    , 0 // venomous // not venomous, but poisonous
    , 0 // fins
    , 4 // legs
    , 1 // tail
    , 1 // domestic
    , 1 // catsize (seems to mean "at least as large as a cat")
    ;
  return x;
}

Vec<Nin> x_woodpecker() {
  Vec<Nin> x;
  x << 0 // hair
    , 1 // feathers
    , 1 // eggs
    , 0 // milk
    , 1 // airborne
    , 0 // aquatic
    , 1 // predator
    , 0 // toothed
    , 1 // backbone
    , 1 // breathes // many salamanders do have lungs, however
    , 0 // venomous // not venomous, but poisonous
    , 0 // fins
    , 2 // legs
    , 1 // tail
    , 0 // domestic
    , 0 // catsize (seems to mean "at least as large as a cat")
    ;
  return x;
}
