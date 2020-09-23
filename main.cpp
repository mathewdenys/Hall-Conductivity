/*
 * Calculates the Hall conductivity as a function of frequency, at a given temperature. This is achieved 
 * by first determining the optimal pairing potential amplitudes by minimising the Helmholtz free energy,
 * which is then passed to the Hall conductivity. This file deals exclusively with the "triplet-triplet"
 * model outlined in my MSc thesis. Refer to: http://mathew.denys.nz/assets/files/2020/msc_thesis.pdf.
 * 
 * Each of FreeEnergy(), OptimizeFreeEnergy(), and HallConductivity() have been confirmed to give results
 * consistent with my pre-existing Julia code, for N=20 and N=50. The output of OptimizeFreeEnergy() is in
 * the least agreement, primarily because it is dependent on the implementation of the Nelder-Mead method.
 * I would guess that the (relatively small) discrepancy arises primarily from the termination condition:
 * I have implemented a rather naive domain termination condition, while the Optim.jl library defaults to
 * a function-value termination condition. On top of this, the output will be somewhat dependent on actual
 * convergence criterion used in the termination condition (taken here to be 1e-6).
 */

#define _USE_MATH_DEFINES   // access M_Pi in <cmath>
#include <iostream>         // std::cout
#include <iomanip>          // std::setw
#include <cmath>            // M_Pi, exp, tanh, sqrt
#include <array>            // std::array
#include <vector>           // std::vector
#include <fstream>          // std::ofstream
#include <sstream>          // std::stringstream
#include <algorithm>        // std::transform
#include <complex>          // for complex numbers
#include "Eigen/Dense"      // for dealing with matrices, vectors, and eigenvalues / eigenvectors

using namespace std::literals::complex_literals; // use literal suffixes {i, if, il} to denote imaginary numbers
using complexd = std::complex<double>;
using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::Vector4cd;

// Define parameters for the normal state
const double t1  = 1.0;          // The first hopping parameter
const double t2  = 0.8*t1;       // The second hopping parameter
const double t3  = 0.1*t1;       // The third hopping parameter
const double mu  = t1;           // The chemical potential
const double soc = 0.25*t1;      // The spin-orbital coupling strength

// Define interaction strengths
const double lambda1 = -0.2;     // The interaction strength in the 01 channel
const double lambda2 = -0.265;   // The interaction strength in the 31 channel
const double lambda3 =  0.03;    // The inter-channel interaction strength

// Define calculation parameters
const double temp      = 0.01;   // The temperature
const int    Nk        = 50;     // The number of momentum points (per dimension) to sum over
const int    Nf        = 150;    // The number of frequencies to evaluate the Hall conductivity at
const double deltaFreq = 0.01;   // The spacing between frequency values (first frequency is deltaFreq)
const double zeroPlus  = 0.001;  // The numerical approximation of the positive infintessimal in the analytic continuation

// kron() calculates the Kronecker product between two matrices
Matrix4cd kron(const Matrix2cd& A, const Matrix2cd& B)
{      
    int p {2};
    int q {2};

    Matrix4cd C;
    for (int i=0;i<2;i++) // Using https://mathworld.wolfram.com/KroneckerProduct.html, but starting indexing from 0 rather than 1
        for (int j=0;j<2;j++)
            for (int k=0;k<2;k++)
                for (int l=0;l<2;l++)
                    C(p*i+k,q*j+l) = A(i,j) * B(k,l);
    return C;
}

// DeltaWrap stores the two (real-valued) elements of the pairing potential
class DeltaWrap
{
public:
    double delta01;
    double delta31;

    DeltaWrap()                       : delta01{ 0.0 }, delta31{ 0.0 } {}
    DeltaWrap(double d01, double d31) : delta01{ d01 }, delta31{ d31 } {}   
};

template <typename T>
DeltaWrap operator*(T scalar, const DeltaWrap& dwin)            { return DeltaWrap(dwin.delta01*scalar, dwin.delta31*scalar); }
DeltaWrap operator+(const DeltaWrap& dw1, const DeltaWrap& dw2) { return DeltaWrap(dw1.delta01 + dw2.delta01, dw1.delta31 + dw2.delta31); }
DeltaWrap operator-(const DeltaWrap& dw1, const DeltaWrap& dw2) { return DeltaWrap(dw1.delta01 - dw2.delta01, dw1.delta31 - dw2.delta31); }

double deltaAbs(DeltaWrap& dw) { return sqrt(pow(dw.delta01,2) + pow(dw.delta31,2)); } // takes the absolute value of a pairing potential

std::ostream& operator<<(std::ostream& out, const DeltaWrap& dw)
{
    out << "[" << dw.delta01 << ", " << dw.delta31 << "]";
    return out;
}

// FreeEnergyParameters stores parameters to be passed to the FreeEnergy() function
struct FreeEnergyParameters
{
    double temp     {}; // The temperature
    int    Nk       {};	// The number of lattice points (per dimension) to sum over
};

std::ostream& operator<<(std::ostream& out, const FreeEnergyParameters& fep)
{
    out << "T=" << fep.temp  << "_Nk=" << fep.Nk;
    return out;
}

// HallParameters stores parameters to be passed to the HallConductivity() function
struct HallParameters
{
    double temp       {};   // The temperature
    int    Nk         {};   // The number of lattice points (per dimension) to sum over
    int    Nf         {};   // The number of frequencies to evaluate the Hall conductivity at
    double deltaFreq  {};   // The spacing between frequency values (first frequency is deltaFreq)
    double zeroPlus   {};   // The numerical approximation of the positive infintessimal in the analytic continuation
};

std::ostream& operator<<(std::ostream& out, const HallParameters& hp)
{
    out << "T="         << hp.temp
        << "_Nk="        << hp.Nk
        << "_Nf="        << hp.Nf
        << "_deltaFreq=" << hp.deltaFreq
        << "_zeroPlus="  << hp.zeroPlus;
    return out;
}

// PauliMatrixGen provides functions that return each of the Pauli matrices (including the identity)
class PauliMatrixGen
{
private:
    Matrix2cd makeMatrix(const complexd& val00, const complexd& val01, const complexd& val10, const complexd& val11)
    {
        Matrix2cd matrix;
        matrix << val00, val01, val10, val11;
        return matrix;
    }

public:
    Matrix2cd makePauli0() { return Matrix2cd::Identity(); }
    Matrix2cd makePauli1() { return makeMatrix(0,  1,  1,  0); }
    Matrix2cd makePauli2() { return makeMatrix(0, -1i, 1i, 0); }
    Matrix2cd makePauli3() { return makeMatrix(1,  0,  0, -1); }
};

// HamiltonianGen provides functions that return the BdG Hamiltonian and each component of the velocity matrix
// These are only constructed as 4x4 matrices in the spin-up sector (not full 8x8 matrices)
class HamiltonianGen
{
public:
    Matrix4cd makeHBdG(const DeltaWrap& amplitudes, double kx, double ky)
    {
        double h00 = make_h00(kx,ky);
        double h10 = make_h10(kx,ky);
        double h30 = make_h30(kx,ky);
        double h23 = make_h23(kx,ky);

        complexd d01 = make_d01(amplitudes.delta01,kx,ky);
        complexd d31 = make_d31(amplitudes.delta31,kx,ky);

        Matrix4cd HBdg;
        HBdg << h00+h30, h10-1i*h23, d01+d31, 0,
                h10+1i*h23, h00-h30, 0, d01-d31,
                std::conj(d01)+std::conj(d31), 0, -h00-h30, -h10+1i*h23,
                0, std::conj(d01)-std::conj(d31), -h10-1i*h23, -h00+h30;
        
        return HBdg;
    }

    Matrix4cd makeVx(double kx, double ky)
    {
        PauliMatrixGen p;
        return make_v00x(kx,ky)*kron(p.makePauli0(),p.makePauli0())
             + make_v10x(kx,ky)*kron(p.makePauli0(),p.makePauli1())
             + make_v30x(kx,ky)*kron(p.makePauli0(),p.makePauli3());
    }

    Matrix4cd makeVy(double kx, double ky)
    {
        PauliMatrixGen p;
        return make_v00y(kx,ky)*kron(p.makePauli0(),p.makePauli0())
             + make_v10y(kx,ky)*kron(p.makePauli0(),p.makePauli1())
             + make_v30y(kx,ky)*kron(p.makePauli0(),p.makePauli3());
    }
    
private:    
    double make_h00(double kx, double ky) {return  -t1*(cos(kx)+cos(ky)) - mu;}
    double make_h10(double kx, double ky) {return 2*t3*(sin(kx)*sin(ky));}
    double make_h30(double kx, double ky) {return  -t2*(cos(kx)-cos(ky));}
    double make_h23(double kx, double ky) {return soc;}

    complexd make_d01(double amplitude, double kx, double ky) {return amplitude*(sin(kx)+1i*sin(ky));}
    complexd make_d31(double amplitude, double kx, double ky) {return amplitude*(sin(kx)-1i*sin(ky));}

    double make_v00x(double kx, double ky) { return   t1*sin(kx);}
    double make_v00y(double kx, double ky) { return   t1*sin(ky);}
    double make_v10x(double kx, double ky) { return 2*t3*cos(kx)*sin(ky);}
    double make_v10y(double kx, double ky) { return 2*t3*sin(kx)*cos(ky);}
    double make_v30x(double kx, double ky) { return   t2*sin(kx);}
    double make_v30y(double kx, double ky) { return  -t2*sin(ky);}
};

// DeltaConversion() converts variational parameters to true pairing potentials
DeltaWrap DeltaConversion(const DeltaWrap& dw)
{
    return DeltaWrap{dw.delta01 + dw.delta31*lambda3/lambda2, dw.delta31 + dw.delta01*lambda3/lambda1};
}

// FreeEnergy() returns the Helmholtz free energy for a given pairing state
double FreeEnergy(const DeltaWrap& delta0optim, FreeEnergyParameters& params)
{ // Note that delta0optim passed in here is *not* the pairing potential, but rather the "unconverted" variational parameters
    double temp = params.temp;
    int    N    = params.Nk;

    double freeEnergySum {}; // to store the momentum dependent part of free energy

    // sum over only one eighth of the brillouin zone (note upper limit in each for loop)
    for (int nx=N/2; nx<N; ++nx)
        for (int ny=N/2; ny<=nx; ++ny) 
        {
            // calculate momentum values
            double kx = (2*nx-N+1)*M_PI/N;
            double ky = (2*ny-N+1)*M_PI/N;

            // don't double count diagonals of the Brillouin zone
            double bzdiagonal = (nx==ny) ? 0.5 : 1.0;

            // explicitly construct one sector of the BdG Hamiltonian (note the pairing potential is converted)
            HamiltonianGen h;
            Matrix4cd HBdg { h.makeHBdG(DeltaConversion(delta0optim),kx,ky) };

            // calculate eigenvalues and eigenvectors of BdG Hamiltonian
            Eigen::ComplexEigenSolver<Matrix4cd> ces;
            ces.compute(HBdg,false); // "false" means compute() only calculate eigenvalues
            
            for (int i=0; i<4; i++)
			{
				double    evali = real(ces.eigenvalues()[i]);
				if (evali > 0) // Only sum over positive eigenvalues. Eigen does not sort eigenvalues in any particular order
					freeEnergySum += (evali + 2*temp*log(1+exp(-evali/temp)))*bzdiagonal;
			}
        }

    return -8*freeEnergySum/pow(N,2) - 0.5*(pow(delta0optim.delta01,2)/lambda1 + pow(delta0optim.delta31,2)/lambda2
                + lambda3/(lambda1*lambda2) * 2 * delta0optim.delta01 * delta0optim.delta31); 
}

// FreeEnergyValue stores a pairing potential along with the corresponding free energy
class FreeEnergyValue
{
public:
    FreeEnergyValue(DeltaWrap x_in, FreeEnergyParameters& params_in) :
    	x{ x_in },
    	params{ params_in },
    	f{ FreeEnergy(x_in,params_in) }
	{ }

    double    getf() { return f; }
    DeltaWrap getx() { return x; }

    void set(DeltaWrap x_in)
    {
        x = x_in;
        f = FreeEnergy(x_in,params); // keep internal state consistent
    }

private:
    DeltaWrap x;					// The pairing potential ("point in 2D Delta-space") at which the free energy is calculated
    double    f;					// The free energy of the system at x
    FreeEnergyParameters params; 	// The parameters required to calcualte the free energy
};

bool operator<(FreeEnergyValue& in1, FreeEnergyValue& in2) { return in1.getf() < in2.getf(); } // for ordering in OptimizeFreeEnergy()

DeltaWrap computeReflectPoint (DeltaWrap& c, DeltaWrap& xh,  double alpha) { return c + alpha*(c-xh); }
DeltaWrap computeExpandPoint  (DeltaWrap& c, DeltaWrap& xr,  double gamma) { return c + gamma*(xr-c); }
DeltaWrap computeContractPoint(DeltaWrap& c, DeltaWrap& xhr, double beta)  { return c + beta*(xhr-c); }

// NMOutput stores useful values and information from running OptimizeFreeEnergy()
struct NMOutput
{
    bool      success;         // 0 if the optimization failed (reached maximum number of iterations), 1 if optimization succeeded (reached termination condition)
    DeltaWrap minimizer;       // The pairing potential that minimizes the free energy
    double    minimum;         // The corresponding minimum value of the free energy
    int       iterations;      // The number of iterations the NM algorithm went through to obtain the output
    double    terminationCond; // The domain termination condition that was used
};

std::ostream& operator<<(std::ostream& out, const NMOutput& output)
{
    std::string success = output.success ? "success" : "failure";
    out << "\n\nThe optimisation was a " << success << ", with the following:\n\n"
        << std::setw(15) << std::left << "\tMinimizer:"   << output.minimizer  << '\n'
        << std::setw(15) << std::left << "\tMinimum:"     << output.minimum    << "\n\n"
        << std::setw(15) << std::left << "\tIterations:"  << output.iterations << '\n'
        << std::setw(15) << std::left << "\tTerm. cond.:" << output.terminationCond;
    return out;
}

using FreeEnergyValuesArray = std::vector<FreeEnergyValue>;

// OptimizeFreeEnergy() finds the pairing potential which minimizes the Free energy using a Nelder Mead algorithm
// Implemented following http://www.scholarpedia.org/article/Nelder-Mead_algorithm. No effort has been made to implement
// a general Nelder-Mead method. OptimizeFreeEnergy() is limited to f: R^2 -> R, and explicitly calls FreeEnergy().
NMOutput OptimizeFreeEnergy(DeltaWrap initialGuess,  FreeEnergyParameters& params) 
{
    const int n               = 2;    // The dimensionality of the input
    double    terminationCond = 1e-6; // The domain termination condition
    const int maxIterations   = 200;  // The maximum number of iterations of the Nelder Mead method before failing
    double    stepsize        = 1.0;  // The step size, h, for constructing the intial right-angled simplex

    // standard parameters for the transformations
    double alpha = 1.0;
    double beta  = 0.5;
    double gamma = 2.0;
    double delta = 0.5;

    // Construct initial right-angled working simplex
    FreeEnergyValuesArray values;
    values.reserve(n+1);
    values.push_back(FreeEnergyValue(initialGuess, params));
    values.push_back(FreeEnergyValue(initialGuess + DeltaWrap(stepsize,0.0), params));
    values.push_back(FreeEnergyValue(initialGuess + DeltaWrap(0.0,stepsize), params));

    // terminationTest() returns true if the domain termination condition has been met (i.e. all points are sufficiently close)
    auto terminationTest = [&](FreeEnergyValuesArray& values)
                            { 
                                return deltaAbs(values.at(0).getx() - values.at(1).getx()) < terminationCond
                                    && deltaAbs(values.at(0).getx() - values.at(2).getx()) < terminationCond
                                    && deltaAbs(values.at(1).getx() - values.at(2).getx()) < terminationCond;
                            };

    // The following loop continues improving the simplex until either
    // 1) the domain termination test is satisfied ( this is preferred over a function test as we want an accurate input value), or
    // 2) it exceeds the specified maximum number of iterations
    int iterations = 0;
    while (!terminationTest(values) && iterations < maxIterations)
    {
        ++iterations;                                                                   // iterate at start because continue statements are used below

        std::sort(values.begin(),values.end(),
                    [](FreeEnergyValue& in1, FreeEnergyValue& in2) {return (in1<in2);}  // sort values such that the first [last] element has the lowest [highest] free energy
                 ); 
        double fLowest  = values.at(0).getf();
        double fSecond  = values.at(1).getf();
        double fHighest = values.at(2).getf();

        // perform the reflection
        DeltaWrap c = 0.5*(values.at(0).getx() + values.at(1).getx());                  // calculate the centroid
        DeltaWrap xReflect = computeReflectPoint(c,values.back().getx(),alpha);
        FreeEnergyValue reflectValue {xReflect,params};
        double fReflect = reflectValue.getf();

        // loopReplace() replaces the worst point in the simplex
        auto loopReplace = [&](FreeEnergyValue& rep) { values.back() = rep; };

        if (fLowest <= fReflect && fReflect < fSecond)
        {
            loopReplace(reflectValue);
            continue;
		}
		
        if (fReflect < fLowest)
        {
            DeltaWrap xExpand = computeExpandPoint(c,reflectValue.getx(),gamma);
            FreeEnergyValue expandValue {xExpand,params};
            double fExpand = expandValue.getf();
            if (fExpand < fReflect)
                loopReplace(expandValue);
            else
                loopReplace(reflectValue);
            continue;
        }
        
        if (fSecond <= fReflect)
        {
            DeltaWrap xContract;
            if (fReflect < fHighest) // contract outside
            {
                xContract = computeContractPoint(c,xReflect,gamma);
                FreeEnergyValue contractValue {xContract,params};
                double fContract = contractValue.getf();
                if (fContract <= fReflect)
                {
                    loopReplace(contractValue);
				    continue;
                }
            }
            
            if (fReflect >= fHighest) // contract inside
            {
                xContract = computeContractPoint(c,values.back().getx(),gamma);
                FreeEnergyValue contractValue {xContract,params};
                double fContract = contractValue.getf();
                if (fContract < fHighest)
                {
                    loopReplace(contractValue);
				    continue;
                }
            }
        }
        
        // if no reflection, contraction, or expansion was performed, perform a shrink
		for (int i=1; i<=2; i++) // update the worst and second worst points
		{
			DeltaWrap x_lowest  = values.front().getx();
			DeltaWrap x_current = values.at(i).getx();
			DeltaWrap x_updated = x_lowest + delta*(x_current - x_lowest);
			values.at(i) = FreeEnergyValue {x_updated,params};
		}
    }

    NMOutput output{ iterations < maxIterations, values.back().getx(), values.back().getf(), iterations, terminationCond };
    return output;
}

// HallConductivity() calculates the Hall conductivity as a function of frequency and overwrites it to hallOut
// Note that delta0optim passed in here is the true pairing potential (unlike in FreeEnergy())
void HallConductivity(std::array<complexd,Nf>& hallOut, const DeltaWrap& delta0optim, HallParameters& params)
{ 
    // local variables for the HallParameters that are used more than once
    const int& temp = params.temp;
    const int& N    = params.Nk;
    assert(N%2==0 && "N must be divisible by 2 in order to sum over partial Brillouin zone");

    // ensure the Hall conductivity array is set to zero before beginning
    for (int freqInd = 0; freqInd < params.Nf; freqInd++)
        hallOut[freqInd] = 0i;
    
    // sum over one eighth of the brillouin zone (note upper limit in each for loop)
    for (int nx=N/2; nx<N; ++nx)
        for (int ny=N/2; ny<=nx; ++ny) 
        {
            // calculate momentum values
            double kx = (2*nx-N+1)*M_PI/N;
            double ky = (2*ny-N+1)*M_PI/N;

            // don't double count diagonals of the Brillouin zone
            double bzdiagonal = (nx==ny) ? 0.5 : 1.0;

            // construct velocity matrices and Hamiltonian in one sector
            HamiltonianGen h;
            Matrix4cd Vx   { h.makeVx(kx,ky) };
            Matrix4cd Vy   { h.makeVy(kx,ky) };
            Matrix4cd HBdg { h.makeHBdG(delta0optim,kx,ky) };

            // calculate eigenvalues and eigenvectors of BdG Hamiltonian
            Eigen::ComplexEigenSolver<Matrix4cd> ces;
            ces.compute(HBdg); 

            for (int freqInd = 0; freqInd < params.Nf; freqInd++)
            {
                complexd omega = (freqInd+1)*params.deltaFreq + 1i*params.zeroPlus;
                for (int i = 0; i < 4; i++)
                {
                    double    evali = real(ces.eigenvalues()[i]);	// eigenvalues() is an accessor only; the computation is done in ces.compute()
                    Vector4cd eveci = ces.eigenvectors().col(i);    // column k of .eigenvectors() corresponds to the kth .eigenvalue()
                    for (int j = 0; j < 4; j++)
                    {
                        double    evalj = real(ces.eigenvalues()[j]);
                        Vector4cd evecj = ces.eigenvectors().col(j);
                        hallOut[freqInd] += 2i*imag(eveci.dot(Vx*evecj) * evecj.dot(Vy*eveci))
                                             * (tanh(evali/(2*temp))-tanh(evalj/(2*temp))) * bzdiagonal
                                             / ( real(omega)*(evali - evalj + omega) ) ;
                    }
                }
            }
        }

    std::transform(hallOut.begin(), hallOut.end(), hallOut.begin(),
                    [&N](const complexd& hall) { return hall*8.0*2.0/(8i*pow(N,2)); } // x8 for BZ summation; x2 for each sector; divide by prefactor
                    );
}

int main()
{
    // Determine optimal pairing potential at the given temperature
    DeltaWrap delta0optim{ 0.01, -0.01 }; // inital guess
    FreeEnergyParameters fparams { temp, Nk };
    std::cout << "\nOptimizing free energy using the following parameters:\n\n\t" << fparams;
    NMOutput freeEnergyOutput = OptimizeFreeEnergy(delta0optim, fparams);
    std::cout << freeEnergyOutput;

    if (freeEnergyOutput.success)
    {
        // Calculate Hall conductivity as a function of frequency
        HallParameters hparams {temp, Nk, Nf, deltaFreq, zeroPlus}; // set up parameters for the calculation
        std::cout << "\n\nCalculating the Hall conductivity using the following parameters:\n\n\t" << hparams;
        std::array<complexd,Nf> hall {};                            // initialize array to store the Hall conductivity
        HallConductivity(hall, delta0optim, hparams);

        // Name the file to write the data to
        std::stringstream ss;
        std::string fileName;
        ss << "hall_" << hparams << ".csv";
        ss >> fileName;

        // Save data to file (file will be overwritten)
        std::ofstream outFile {fileName};

        if (!outFile)
        {
            std::cerr << fileName <<" could not be opened for writing";
            return 1;
        }
        
        int counter = 0;
        for (complexd cnum : hall)
        {
            outFile << counter*hparams.deltaFreq << ","  // column 1: the frequency
                    << real(cnum) << ","                // column 2: real part of Hall the conductivity
                    << imag(cnum) <<"\n";               // column 3: imaginary part of the Hall conductivity
            counter++;
        }

        outFile.close();
        std::cout << "\n\nHall conductivity data has been saved to " << fileName;
    }
    else
    {
        std::cout << "\n\nHall conductivity has not been calculated.";
    }

    return 0;
}