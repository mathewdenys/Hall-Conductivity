#define _USE_MATH_DEFINES
#include <iostream> // std::cout
#include <cmath>    // M_Pi, exp
#include <array>    // std::array
#include <fstream>  // std::ofstream
#include <sstream>  // std::stringstream
#include <complex>
#include "Eigen/Dense"

using namespace std::literals::complex_literals; // Use literal suffix i (or if, il) to denote an imaginary number
using complexd = std::complex<double>;
using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::Vector4cd;
using std::cout;

// Define parameters for the normal state
const double t1  = 1.0;
const double t2  = 0.8*t1;
const double t3  = 0.1*t1;
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
    // implemented using formula from mathworld (https://mathworld.wolfram.com/KroneckerProduct.html)
    // using the notation used there, but starting indexing from 0 rather than 1

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

struct DeltaWrap
{
    double val[2]; // the pairing potential can be real valued with no loss of generality
};

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
    double make_v30y(double kx, double ky) {return   -t2*sin(ky);}
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
            for (int i=1; i<=4; i++)
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
    const int    &N    = params.Nk;
    const int    &temp = params.temp;
    assert(N%2==0 && "N must be divisible by 2 in order to sum over partial Brillouin zone");
    
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
                    double    evali = real(ces.eigenvalues()[i]);
                    Vector4cd eveci = ces.eigenvectors().col(i);
                    for (int j = 0; j < 4; j++)
                    {
                        double    evalj = real(ces.eigenvalues()[j]);
                        Vector4cd evecj = ces.eigenvectors().col(j);
                        hallOut[freqInd] += 2i*imag(eveci.dot(Vx*evecj) * evecj.dot(Vy*eveci))
                                             * (tanh(evali/(2*temp))-tanh(evalj/(2*temp)))
                                             / ( real(omega)*(evali - evalj + omega) )
                                             * bzdiagonal * 8.0 * 2.0 /(8i*pow(N,2)); // x8 for BZ summation; x2 for each sector; divide by prefactor
                    }
                }
            }
        }
}

int main()
{
    // Define pairing potential. In the future this will be calculated
    DeltaWrap delta0optim;
    delta0optim.val[0] = 0.2;
    delta0optim.val[1] = 0.05;

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
