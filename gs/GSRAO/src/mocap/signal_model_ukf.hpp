/* 
 * This header defines the system state, system model, measurement and 
 * measurement model of a signal for kalman filtering
 *
 * Autor:
 *      Roice Luo
 * Date:
 *      2016.09.12
 */

#ifndef SIGNAL_MODEL_UKF_HPP
#define SIGNAL_MODEL_UKF_HPP

#include <UnscentedKalmanFilter.hpp>

namespace Signal_Model
{
    /**
     * @brief System state vector-type for a signal
     * 
     * This is s sytem state for a signal that is characterized by its
     * output value, [reading, reading']
     *
     * @param T Numeric scalar type
     */
template<typename T>
class State : public Kalman::Vector<T, 2> 
{
public:
    KALMAN_VECTOR(State, T, 2)
    
    //! sensor output value
    static constexpr size_t X = 0;
    //! first derivative of sensor value
    static constexpr size_t dX = 1;
    
    T reading()       const { return (*this)[ X ]; }
    T dreading()       const { return (*this)[ dX ]; }
    
    T& reading()      { return (*this)[ X ]; }
    T& dreading()      { return (*this)[ dX ]; }
};

template<typename T>
class Control : public Kalman::Vector<T, 0> 
{
public:
    KALMAN_VECTOR(Control, T, 0)
};

/**
 * @brief System model for a signal
 *
 * This is the system model defining how the output changes from one
 * time-step to the next, i.e. how the system state evolves over time.
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class SystemModel : public Kalman::SystemModel<State<T>, Control<T>, CovarianceBase> 
{
public:
    //! State type shortcut definition
    typedef State<T> S;

    //! Control type shortcut definition
    typedef Control<T> C; // not used

    // delta t
    float dt = 1.0/50.0;    // 50 Hz microbee control loop
    
    /**
     * @brief Definition of (non-linear) state transition function
     *
     * This function defines how the system state is propagated through time,
     * i.e. it defines in which state \f$\hat{x}_{k+1}\f$ is system is expected to 
     * be in time-step \f$k+1\f$ given the current state \f$x_k\f$ in step \f$k\f$ and
     * the system control input \f$u\f$.
     *
     * @param [in] x The system state in current time-step
     * @param [in] u not used
     * @returns The (predicted) system state in the next time-step
     */
    S f(const S& x, const C& u) const
    {
        //! Predicted state vector after transition
        S x_;
        
        // New reading given by old reading plus reading change
        x_.reading() = x.reading() + x.dreading()*dt;
        x_.dreading() = x.dreading();
        
        // Return transitioned state vector
        return x_;
    }
};

/**
 * @brief Measurement vector measuring sensor output (i.e. by using ADC)
 *
 * @param T Numeric scalar type
 */
template<typename T>
class Measurement : public Kalman::Vector<T, 1> 
{
public:
    KALMAN_VECTOR(Measurement, T, 1)
    
    //! sensor output
    static constexpr size_t Z = 0;
    
    T z()  const { return (*this)[ Z ]; }
    T& z() { return (*this)[ Z ]; }
};

/**
 * @brief Measurement model for measuring sensor output
 *
 * This is the measurement model for measuring the output of
 * the MOX sensor. This could be realized by an ADC.
 *
 * @param T Numeric scalar type
 * @param CovarianceBase Class template to determine the covariance representation
 *                       (as covariance matrix (StandardBase) or as lower-triangular
 *                       coveriace square root (SquareRootBase))
 */
template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class MeasurementModel : public Kalman::MeasurementModel<State<T>, Measurement<T>, CovarianceBase>
{
public:
    //! State type shortcut definition
    typedef State<T> S;
    
    //! Measurement type shortcut definition
    typedef Measurement<T> M;
    
    MeasurementModel()
    {
    }
    
    /**
     * @brief Definition of (possibly non-linear) measurement function
     *
     * This function maps the system state to the measurement that is expected
     * to be received from the sensor assuming the system is currently in the
     * estimated state.
     *
     * @param [in] x The system state in current time-step
     * @returns The (predicted) sensor measurement for the system state
     */
    M h(const S& x) const
    {
        M measurement;
        
        // H = [1 0]
        measurement.z() = x.reading();
        
        return measurement;
    }
};
} // namespace Signal_Model

#endif
