#define _USE_MATH_DEFINES
#include <cmath>    // M_Pi
#include <iostream> // std::cout
#include <complex>
#include "Eigen/Dense"

// Use literal suffix i (or if, il) to denote an imaginary number
// Why does "using std::literals::complex_literals::operator""i;" throw a compiler warning?
using namespace std::literals::complex_literals;
using complexd = std::complex<double>; // easier to read, and if I need to change to e.g. long double through it will be easier
using std::cout;
using Eigen::Matrix2cd;
using Eigen::Matrix4cd;

struct DeltaWrap
{
    complexd val[2];
};

class PauliMatrixGen
{
public:
    Eigen::Matrix2cd makepauli0()
    {
        return Eigen::Matrix2cd::Identity();
    }
    Eigen::Matrix2cd makepauli1()
    {
        Eigen::Matrix2cd pauli1;
        pauli1 << 0, 1, 1, 0;
        return pauli1;
    }
    Eigen::Matrix2cd makepauli2()
    {
        Eigen::Matrix2cd pauli2;
        pauli2 << 0, -1i, 1i, 0;
        return pauli2;
    }
    Eigen::Matrix2cd makepauli3()
    {
        Eigen::Matrix2cd pauli3;
        pauli3 << 1, 0, 0, -1;
        return pauli3;
    }
};

int hallconductivity(DeltaWrap delta0optim, int Nmax, double temp, int omegamax, double deltaomega, double zeroplus);
DeltaWrap deltaconversion(DeltaWrap delta);
Matrix4cd kron(Matrix2cd A, Matrix2cd B);

// Define parameters for the normal state
// Consider other options other than making these global
const double t1 {1.0};
const double t2 {0.8*t1};
const double t3 {0.1*t1};
const double mu {t1};
const double soc{0.25*t1};

// Define interaction strengths
// Consider other options other than making these global
const double lambda1 {-0.2};
const double lambda2 {-0.265};
const double lambda3 {0.03};

int main()
{
    DeltaWrap delta0optim; // in the future this will be determined (as a function of T), but for now I will assign it a value
    delta0optim.val[0] = 0.2;
    delta0optim.val[1] = 0.05;

    // call Hall conductivity function

    return 0;
}

DeltaWrap deltaconversion(DeltaWrap delta)
{
    DeltaWrap delta_converted;
    delta_converted.val[0] = delta.val[0]+delta.val[1]*lambda3/lambda2;
    delta_converted.val[1] = delta.val[1]+delta.val[0]*lambda3/lambda1;
    return delta_converted;
}

int hallconductivity(DeltaWrap delta0optim, int Nmax, double temp, int omegamax, double deltaomega, double zeroplus)
{
    // TODO: INITIALIZE ARRAY TO STORE HALL CONDUCTIVITY (USE STD::VECTOR?)
    // Bear in mind you can't return an array
    
    for (int nx=Nmax/2; nx<Nmax; ++nx) {
        for (int ny=Nmax/2; ny<Nmax; ++ny) {

            // calculate momentum values
            double kx {(2*nx-Nmax+1)*M_PI/Nmax};
            double ky {(2*ny-Nmax+1)*M_PI/Nmax};

            // don't double count diagonals of the Brillouin zone
            double bzdiagonal = (nx==ny) ? 0.5 : 1.0;

            // make Pauli matrices
            PauliMatrixGen p;
            Eigen::Matrix2cd pauli0 {p.makepauli0()};
            Eigen::Matrix2cd pauli1 {p.makepauli1()};
            Eigen::Matrix2cd pauli2 {p.makepauli2()};
            Eigen::Matrix2cd pauli3 {p.makepauli3()};

            // construct normal state Hamiltonian
            // these are all real-valued
            double h00 { -t1*(cos(kx)+cos(ky)) - mu};
            double h10 {2*t3*(sin(kx)*sin(ky))};
            double h30 { -t2*(cos(kx)-cos(ky))};
            double h23 {soc};

            // construct velocity terms (vx5=vy5=0) in one sector
            double v00x {t1*sin(kx)};
            double v00y {t1*sin(ky)};
            double v10x {2*t3*cos(kx)*sin(ky)};
            double v10y {2*t3*sin(kx)*cos(ky)};
            double v30x {t2*sin(kx)};
            double v30y {t2*sin(ky)};
            Matrix4cd Vx {v00x*kron(pauli0,pauli0) + v10x*kron(pauli0,pauli1) + v30x*kron(pauli0,pauli3)};
            Matrix4cd Vy {v00y*kron(pauli0,pauli0) + v10y*kron(pauli0,pauli1) + v30y*kron(pauli0,pauli3)};
            
            // construct pairing potential
            DeltaWrap delta0optim_converted = deltaconversion(delta0optim);
            complexd d01 = delta0optim_converted.val[0]*(sin(kx)+1i*sin(ky));
            complexd d31 = delta0optim_converted.val[1]*(sin(kx)-1i*sin(ky));

            // explicitly construct & calculate evals of one sector of the BdG Hamiltonian
            Matrix4cd HBdg;
            HBdg << h00+h30, h10-1i*h23, d01+d31, 0,
                    h10+1i*h23, h00-h30, 0, d01-d31,
                    std::conj(d01)+std::conj(d31), 0, -h00-h30, -h10+1i*h23,
                    0, std::conj(d01)-std::conj(d31), -h10-1i*h23, -h00+h30;
            
            Eigen::ComplexEigenSolver<Matrix4cd> ces(HBdg,false);
            Eigen::Vector4cd evals = ces.eigenvalues();
            Eigen::Matrix4cd evecs = ces.eigenvectors();

            // TODO: PERFORM THE LOOP OVER MATSUBARA FREQUENCIES
        }
    }

    return 0; // meaningless for now
}

Matrix4cd kron(Matrix2cd A, Matrix2cd B)
{
    // implemented using formula from mathworld (https://mathworld.wolfram.com/KroneckerProduct.html)
    // using the notation used there (p,q,i,j,k,l)

    int p {2};
    int q {2};

    Matrix4cd C;

    for (int i=1;i<=2;i++)
    {
        for (int j=1;j<=2;j++)
        {
            for (int k=1;k<=2;k++)
            {
                for (int l=1;l<=2;l++)
                {
                    // indexing here starts from 0
                    // but indexing in the formula starts from 1
                    // so I subtract 1 from each index
                    C(p*(i-1)+k-1,q*(j-1)+l-1) = A(i-1,j-1) * B(k-1,l-1);
                }
            }
        }
    }

    return C;

}