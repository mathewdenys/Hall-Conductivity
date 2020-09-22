#define _USE_MATH_DEFINES
#include <iostream> // std::cout
#include <cmath>    // M_Pi, exp
#include <array>    // std::array
#include <vector>   // std::vector
#include <fstream>  // std::ofstream
#include <sstream>  // std::stringstream
#include <algorithm>// std::transform
#include <complex>
#include "Eigen/Dense"

using namespace std::literals::complex_literals; // use literal suffixes {i, if, il} to denote imaginary numbers
using complexd = std::complex<double>;
using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::Vector4cd;
using std::cout;

// Define parameters for the normal state
const double t1  = 1.0;         // hopping parameter #1
const double t2  = 0.8*t1;      // hopping parameter #2
const double t3  = 0.1*t1;      // hopping parameter #3
const double mu  = t1;          // chemical potential
const double soc = 0.25*t1;     // spin-orbital coupling strength

// Define interaction strengths
const double lambda1 = -0.2;    // interaction strength in the 01 channel
const double lambda2 = -0.265;  // interaction strength in the 31 channel
const double lambda3 =  0.03;   // inter-channel interaction strength

// Define calculation parameters
const double T1        = 0.0001;
const double T2        = 0.02;
const int    NT        = 200;
const int    Nk_gap    = 10;
const int    Nk_hall   = 10;
const int    Nf        = 150;
const double deltaFreq = 0.01;
const double zeroPlus  = 0.001;

Matrix4cd kron(Matrix2cd A, Matrix2cd B) // The Kronecker product between two matrices
{
    // Using https://mathworld.wolfram.com/KroneckerProduct.html, but starting indexing from 0 rather than 1

    int p {2};
    int q {2};

    Matrix4cd C;

    for (int i=0;i<2;i++)
        for (int j=0;j<2;j++)
            for (int k=0;k<2;k++)
                for (int l=0;l<2;l++)
                    C(p*i+k,q*j+l) = A(i,j) * B(k,l);

    return C;
}

class DeltaWrap
{
public: 
    double val[2]; // the pairing potential can be real valued with no loss of generality

    DeltaWrap()
    {
        val[0] = 0.0;
        val[1] = 0.0;
    }

    DeltaWrap(double val0, double val1)
    {
        val[0] = val0;
        val[1] = val1;
    }
};

DeltaWrap operator+(const DeltaWrap& dw1, const DeltaWrap& dw2)
{
    return DeltaWrap(dw1.val[0]+dw2.val[0],dw1.val[1]+dw2.val[1]);
}

DeltaWrap operator-(const DeltaWrap& dw1, const DeltaWrap& dw2)
{
    return DeltaWrap(dw1.val[0]-dw2.val[0],dw1.val[1]-dw2.val[1]);
}

template <typename T>
DeltaWrap operator*(T scalar, DeltaWrap const& dwin)
{
    DeltaWrap dwout;
    dwout.val[0] = scalar * dwin.val[0];
    dwout.val[1] = scalar * dwin.val[1];
    return dwout;
}

struct FreeEnergyParameters // Parameters to be passed to the FreeEnergy() function.
{
    double temp     {}; // The temperature
    int Nk          {};	// The number of lattice points per dimension
};

struct HallParameters // Parameters to be passed to the HallConductivity() function
{
    double temp       {};   // The temperature
    int    Nk         {};   // The number of lattice points per dimension
    int    Nf         {};   // The number of frequencies to evaluate the Hall conductivity at
    double deltaFreq  {};   // The spacing between frequencies
    double zeroPlus   {};   // The numerical approximation of the positive infintessimal in the analytic continuation
};

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

class HamiltonianGen
{
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
    // note that v23x = v23y = 0 in this model

public:
    Matrix4cd make_HBdg(DeltaWrap amplitudes, double kx, double ky) // construct one sector of the BdG Hamiltonian
    {
        double h00 = make_h00(kx,ky);
        double h10 = make_h10(kx,ky);
        double h30 = make_h30(kx,ky);
        double h23 = make_h23(kx,ky);

        complexd d01 = make_d01(amplitudes.val[0],kx,ky);
        complexd d31 = make_d31(amplitudes.val[1],kx,ky);

        Matrix4cd HBdg;
        HBdg << h00+h30, h10-1i*h23, d01+d31, 0,
                h10+1i*h23, h00-h30, 0, d01-d31,
                std::conj(d01)+std::conj(d31), 0, -h00-h30, -h10+1i*h23,
                0, std::conj(d01)-std::conj(d31), -h10-1i*h23, -h00+h30;
        
        return HBdg;
    }

    Matrix4cd make_Vx(double kx, double ky)
    {
        PauliMatrixGen p;
        Matrix2cd pauli0 {p.makePauli0()};
        Matrix2cd pauli1 {p.makePauli1()};
        Matrix2cd pauli3 {p.makePauli3()};
        return make_v00x(kx,ky)*kron(pauli0,pauli0) + make_v10x(kx,ky)*kron(pauli0,pauli1) + make_v30x(kx,ky)*kron(pauli0,pauli3);
    }

    Matrix4cd make_Vy(double kx, double ky)
    {
        PauliMatrixGen p;
        Matrix2cd pauli0 {p.makePauli0()};
        Matrix2cd pauli1 {p.makePauli1()};
        Matrix2cd pauli3 {p.makePauli3()};
        return make_v00y(kx,ky)*kron(pauli0,pauli0) + make_v10y(kx,ky)*kron(pauli0,pauli1) + make_v30y(kx,ky)*kron(pauli0,pauli3);
    }
};

DeltaWrap DeltaConversion(const DeltaWrap& delta) // Converts variational parameters to true pairing potentials
{
    DeltaWrap deltaConverted;
    deltaConverted.val[0] = delta.val[0]+delta.val[1]*lambda3/lambda2;
    deltaConverted.val[1] = delta.val[1]+delta.val[0]*lambda3/lambda1;
    return deltaConverted;
}

double FreeEnergy(const DeltaWrap& delta0optim, FreeEnergyParameters& params) // Returns the Helmholtz free energy for a given pairing state
{ // Note that delta0optim passed in here is *not* the pairing potential, but the "unconverted" variational parameters
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
            Matrix4cd HBdg { h.make_HBdg(DeltaConversion(delta0optim),kx,ky) };

            // calculate eigenvalues and eigenvectors of BdG Hamiltonian
            Eigen::ComplexEigenSolver<Matrix4cd> ces;
            ces.compute(HBdg,false); // false -> only calculate eigenvalues
            
            // sum over positive eigenvalues
            for (int i=0; i<4; i++)
			{
				double    evali = real(ces.eigenvalues()[i]);
				if (evali > 0) // Eigen does not sort eigenvalues in any particular order
					freeEnergySum += (evali + 2*temp*log(1+exp(-evali/temp)))*bzdiagonal;
			}
        }

    return -8*freeEnergySum/pow(N,2) - 0.5*(pow(delta0optim.val[0],2)/lambda1 + pow(delta0optim.val[1],2)/lambda2
                + lambda3/(lambda1*lambda2) * 2 * delta0optim.val[0] * delta0optim.val[1]); 
}

void HallConductivity(std::array<complexd,Nf>& hallOut, const DeltaWrap& delta0optim, HallParameters& params)
{ // Note that delta0optim passed in here *is* the true pairing potential (unlike in FreeEnergy())
    // local variables for the HallParameters that are used more than once
    const int& N    = params.Nk;
    const int& temp = params.temp;
    assert(N%2==0 && "N must be divisible by 2 in order to sum over partial Brillouin zone");

    // ensure the Hall conductivity array is set to zero before beginning
    for (int freqInd = 0; freqInd < params.Nf; freqInd++)
        hallOut[freqInd] = 0i;
    
    // sum over only one eighth of the brillouin zone (note upper limit in each for loop)
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
            Matrix4cd Vx   { h.make_Vx(kx,ky) };
            Matrix4cd Vy   { h.make_Vy(kx,ky) };
            Matrix4cd HBdg { h.make_HBdg(delta0optim,kx,ky) };

            // calculate eigenvalues and eigenvectors of BdG Hamiltonian
            Eigen::ComplexEigenSolver<Matrix4cd> ces;
            ces.compute(HBdg); // column k of .eigenvectors() corresponds to the kth .eigenvalue()

            for (int freqInd = 0; freqInd < params.Nf; freqInd++)
            {
                complexd omega = (freqInd+1)*params.deltaFreq + 1i*params.zeroPlus;
                for (int i = 0; i < 4; i++)
                {
                    double    evali = real(ces.eigenvalues()[i]);	// Accessor only, compute is done in ces.compute
                    Vector4cd eveci = ces.eigenvectors().col(i);
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
                        [&N](const complexd& hall)
						{
							return hall*8.0*2.0/(8i*pow(N,2)); // x8 for BZ summation; x2 for each sector; divide by prefactor
						});
}





// Below is my implementation of the Nelder-Mead algorithm
// I am limiting its applicability to what I need here
// i.e. minimising a function, f: R^2 -> R (not the general R^n -> R case)
// Implemented following http://www.scholarpedia.org/article/Nelder-Mead_algorithm

class FreeEnergyValue
{

public:

    FreeEnergyValue(DeltaWrap x_in, FreeEnergyParameters& params_in) :
    	x{ x_in},
    	params{ params_in },
    	f{ FreeEnergy(x,params) }
	{ }

    void set_x(DeltaWrap x_in) {x = x_in;}
    void calc_f(double f_in)   {f = FreeEnergy(x,params);} // todo: eventaully get rid of this, so that the internal state is always consistent
    double get_f()    {return f;}
    DeltaWrap get_x() {return x;}


private:
    DeltaWrap x;					///< This is ..
    double f;						///< This is ..
    FreeEnergyParameters params; 	///<This is ..
};

bool operator<(FreeEnergyValue& in1, FreeEnergyValue& in2) // for ordering in OptimizeFreeEnergy()
{
    return in1.get_f() < in2.get_f();
}

DeltaWrap computeReflectPoint(DeltaWrap& c, DeltaWrap& x_highest, double alpha)
{
    DeltaWrap x_reflect = c + alpha*(c-x_highest);
    return x_reflect;
}

DeltaWrap computeExpandPoint(DeltaWrap& c, DeltaWrap& x_reflect, double gamma)
{
    DeltaWrap x_expand = c + gamma*(x_reflect-c);
    return x_expand;
}

DeltaWrap computeContractPoint(DeltaWrap& c, DeltaWrap& x_highest_or_reflect, double beta)
{
    DeltaWrap x_contract = c + beta*(x_highest_or_reflect - c);
    return x_contract;
}

void OptimizeFreeEnergy(DeltaWrap initial_guess, int max_iters, FreeEnergyParameters& params) 
{
	// OptimizeFreeEnergy finds the pairing potential which minimizes the Free energy using a Nelder Meade numerical optimization.
	
    const int n = 2;    // the dimensionality of the input

    // standard parameters for the transformation
    double alpha = 1.0;
    double beta  = 0.5;
    double gamma = 2.0;
    double delta = 0.5;

    // Initialise termination tests to be false
    bool term_x = 0; // domain termination test. Becomes true when working simplex is suficiently small
    bool term_f = 0; // function termination test. Becomes true when some function values are sufficiently close to each other
    bool fail   = 0; // no-convergence test. Becomes true if the number of iterations exceeds max_iters

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
        // 1. Ordering
        std::sort(initialValues.begin(),initialValues.end(), [](FreeEnergyValue& in1, FreeEnergyValue& in2) {return (in1<in2);});

        // 2. Centroid
        // Once sorted, the last element of 'initialValues' corresponds to the highest free energy, so is ignored when determining the centroid
        DeltaWrap c = 0.5*(initialValues.at(0).get_x() + initialValues.at(1).get_x());

        // 3. Transformation
        FreeEnergyValue replacementValue {initialValues.back()}; // initialise value to replace the highest free energy

		auto loopAssign = [&](FreeEnergyValue& rep) { replacementValue = rep; initialValues.back() = rep; }; // replace worst point on the simplex


        DeltaWrap x_highest = initialValues.back().get_x();
        DeltaWrap x_reflect = computeReflectPoint(c,x_highest,alpha);
        FreeEnergyValue reflectValue {x_reflect,params};

        double f_highest = initialValues.at(2).get_f();
        double f_second  = initialValues.at(1).get_f();
        double f_lowest  = initialValues.at(0).get_f();
        
        double f_reflect = reflectValue.get_f();

        if (f_lowest <= f_reflect && f_reflect < f_second)
        {
            loopAssign(reflectValue)
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
    DeltaWrap delta0optim;
    delta0optim.val[0] = 0.0; //0.2;
    delta0optim.val[1] = 0.0; //0.05;

    // Set up parameters for the calculation
    HallParameters params;
    params.temp = T1;
    params.Nk = Nk_hall;
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
