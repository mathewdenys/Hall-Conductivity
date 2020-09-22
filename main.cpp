/*
 * Calculates the Hall conductivity as a function of frequency, at a given temperature. This is achieved 
 * by first determining the optimal pairing potential amplitudes by minimising the Helmholtz free energy,
 * which is then passed to the Hall conductivity. This file deals exclusively with the "triplet-triplet"
 * model outlined in my MSc thesis, using a phenomenological interaction potential.
 * Refer to: http://mathew.denys.nz/assets/files/2020/msc_thesis.pdf
 */

#define _USE_MATH_DEFINES   // access M_Pi in <cmath>
#include <iostream>         // std::cout
#include <cmath>            // M_Pi, exp, tanh
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
const double temp      = 0.0001; // The temperature
const int    Nk        = 10;     // The number of momentum points (per dimension) to sum over
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

// FreeEnergyParameters stores parameters to be passed to the FreeEnergy() function
struct FreeEnergyParameters
{
    double temp     {}; // The temperature
    int    Nk       {};	// The number of lattice points (per dimension) to sum over
};

// HallParameters stores parameters to be passed to the HallConductivity() function
struct HallParameters
{
    double temp       {};   // The temperature
    int    Nk         {};   // The number of lattice points (per dimension) to sum over
    int    Nf         {};   // The number of frequencies to evaluate the Hall conductivity at
    double deltaFreq  {};   // The spacing between frequency values (first frequency is deltaFreq)
    double zeroPlus   {};   // The numerical approximation of the positive infintessimal in the analytic continuation
};

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





// Below is my implementation of the Nelder-Mead algorithm
// I am limiting its applicability to what I need here
// i.e. minimising a function, f: R^2 -> R (not the general R^n -> R case)
// Implemented following http://www.scholarpedia.org/article/Nelder-Mead_algorithm

// FreeEnergyValue stores a pairing potential along with the corresponding free energy
class FreeEnergyValue
{
public:
    FreeEnergyValue(DeltaWrap x_in, FreeEnergyParameters& params_in) :
    	x{ x_in },
    	params{ params_in },
    	f{ FreeEnergy(x,params) }
	{ }

    double    get_f() { return f; }
    DeltaWrap get_x() { return x; }

    void set(DeltaWrap x_in)
    {
        x = x_in;
        f = FreeEnergy(x_in,params); // keep internal state consistent
    }

private:
    DeltaWrap x;					// The pairing potential ("point in 2D Delta-space") at which the free energy is calculates
    double    f;					// The free energy of the system at x
    FreeEnergyParameters params; 	// The parameters required to calcualte the free energy
};

bool operator<(FreeEnergyValue& in1, FreeEnergyValue& in2) { return in1.get_f() < in2.get_f(); } // for ordering in OptimizeFreeEnergy()

DeltaWrap computeReflectPoint (DeltaWrap& c, DeltaWrap& xh,  double alpha) { return c + alpha*(c-xh); }
DeltaWrap computeExpandPoint  (DeltaWrap& c, DeltaWrap& xr,  double gamma) { return c + gamma*(xr-c); }
DeltaWrap computeContractPoint(DeltaWrap& c, DeltaWrap& xhr, double beta)  { return c + beta*(xhr-c); }

// OptimizeFreeEnergy() finds the pairing potential which minimizes the Free energy using a Nelder Mead algorithm
void OptimizeFreeEnergy(DeltaWrap initial_guess, int max_iters, FreeEnergyParameters& params) 
{
    const int n = 2;    // the dimensionality of the input

    // standard parameters for the transformation
    double alpha = 1.0;
    double beta  = 0.5;
    double gamma = 2.0;
    double delta = 0.5;

    // Initialise termination tests to be false
    bool term_x = 0; // Domain termination test. Becomes true when working simplex is suficiently small
    bool term_f = 0; // Function termination test. Becomes true when some function values are sufficiently close to each other
    bool fail   = 0; // No-convergence test. Becomes true if the number of iterations exceeds max_iters

    // Construct initial working simplex (use a right-angled simplex)
    // The FreeEnergyValue() constructor calculates f
    double stepsize = 1.0;
    std::vector<FreeEnergyValue> initialValues;
    initialValues.reserve(n+1);
    initialValues.push_back(FreeEnergyValue(initial_guess, params));
    initialValues.push_back(FreeEnergyValue(initial_guess + DeltaWrap(stepsize,0.0), params));
    initialValues.push_back(FreeEnergyValue(initial_guess + DeltaWrap(0.0,stepsize), params));

    // Minimise the function
    while (!term_x && !term_f && !fail)
    {
        // 1. Ordering (after sorting, the first / last element if initialValues has the lowest / highest free energy)
        std::sort(initialValues.begin(),initialValues.end(), [](FreeEnergyValue& in1, FreeEnergyValue& in2) {return (in1<in2);});

        // 2. Centroid
        DeltaWrap c = 0.5*(initialValues.at(0).get_x() + initialValues.at(1).get_x());

        // 3. Transformation
        FreeEnergyValue replacementValue {initialValues.back()}; // initialise value to replace the highest free energy

		auto loopAssign = [&](FreeEnergyValue& rep) { replacementValue = rep; initialValues.back() = rep; }; // replace worst point on the simplex


        DeltaWrap x_highest = initialValues.back().get_x();
        DeltaWrap x_reflect = computeReflectPoint(c,x_highest,alpha);
        FreeEnergyValue reflectValue {x_reflect,params};

        double f_lowest  = initialValues.at(0).get_f();
        double f_second  = initialValues.at(1).get_f();
        double f_highest = initialValues.at(2).get_f();
        
        double f_reflect = reflectValue.get_f();

        if (f_lowest <= f_reflect && f_reflect < f_second)
        {
            loopAssign(reflectValue);
            continue;
		}
		
        if (f_reflect < f_lowest)
        {
            DeltaWrap x_expand = computeExpandPoint(c,reflectValue.get_x(),gamma);
            FreeEnergyValue expandValue {x_expand,params};
            double f_expand = expandValue.get_f();
            if (f_expand < f_reflect)
                loopAssign(expandValue);
            else
                loopAssign(reflectValue);
            continue;
        }
        
        if (f_second <= f_reflect)
        {
            DeltaWrap x_contract;
            if (f_reflect < f_highest)
            {
                x_contract = computeExpandPoint(c,x_reflect,gamma);
                FreeEnergyValue contractValue {x_contract,params};
                double f_contract = contractValue.get_f();
                if (f_contract <= f_reflect)
                    loopAssign(contractValue);
				continue;
            }
            
            if (f_reflect >= f_highest)
            {
                x_contract = computeExpandPoint(c,x_highest,gamma);
                FreeEnergyValue contractValue {x_contract,params};
                double f_contract = contractValue.get_f();
                if (f_contract < f_highest)
                	loopAssign(contractValue);
				continue;
            }
            loopAssign(replacementValue);
        }
        
		for (int i=1; i<=2; i++) // update the worst and second worst points
		{
			DeltaWrap x_lowest  = initialValues.front().get_x();
			DeltaWrap x_current = initialValues.at(i).get_x();
			DeltaWrap x_updated = x_lowest + delta*(x_current - x_lowest);
			initialValues.at(i) = FreeEnergyValue {x_updated,params};
		}
    }

}





int main()
{
    // Define pairing potential. In the future this will be calculated
    DeltaWrap delta0optim{ 0.2, 0.05};

    // Set up parameters for the calculation
    HallParameters params;
    params.temp = temp;
    params.Nk = Nk;
    params.Nf = Nf;
    params.deltaFreq = deltaFreq;
    params.zeroPlus = zeroPlus;

    // Calculate Hall conductivity
    std::array<complexd,Nf> hall {}; // initialize array to store the Hall conductivity
    HallConductivity(hall, delta0optim, params);

    // Name the file to write the data to
    std::stringstream ss;
    ss  << "hall_T=" << params.temp
        << "_Nk=" << params.Nk
        << "_Nf=" << params.Nf
        << "_deltaFreq=" << params.deltaFreq
        << "_zeroPlus=" << params.zeroPlus
        << ".csv";
    std::string fileName;
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
        outFile << counter*params.deltaFreq << ","  // column 1: the frequency
                << real(cnum) << ","                // column 2: real part of Hall the conductivity
                << imag(cnum) <<"\n";               // column 3: imaginary part of the Hall conductivity
        counter++;
    }

    outFile.close();
    std::cout << "Hall conductivity data has been saved to " << fileName;

    return 0;
}
