/*
 * Serial driver
 *
 * Author:
 *      Lorenz Meier, <lm@inf.ethz.ch>
 *      Roice Luo
 * Date:
 *      2016.05.18
 */
// Serial includes
#include <stdio.h>   /* Standard input/output definitions */
#include <string.h>  /* String function definitions */
#include <unistd.h>  /* UNIX standard function definitions */
#include <fcntl.h>   /* File control definitions */
#include <errno.h>   /* Error number definitions */
#include <termios.h> /* POSIX terminal control definitions */
#ifdef __linux
#include <sys/ioctl.h>
#endif

int serial_open(const char* port)
{
	int fd; /* File descriptor for the port */
	
	// Open serial port
	// O_RDWR - Read and write
	// O_NOCTTY - Ignore special chars like CTRL-C
    // O_NDELAY - not care DCD state
	fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
	if (fd == -1)
		/* Could not open the port. */
		return(-1);
	else
		fcntl(fd, F_SETFL, 0);
	
	return (fd);
}

// 8N1
bool serial_setup(int fd, int baud)
{
	struct termios  options;

    // validate serial port
	if(!isatty(fd)) {
        //fprintf(stderr, "\nERROR: file descriptor %d is NOT a serial port\n", fd);
		return false;
	}
    // Get the current options for the port
	if(tcgetattr(fd, &options) < 0) {
		//fprintf(stderr, "\nERROR: could not read configuration of fd %d\n", fd);
		return false;
	}

    // Baud rate
    switch (baud)
	{
		case 1200:
			if (cfsetispeed(&options, B1200) < 0 || cfsetospeed(&options, B1200) < 0)
			{
				fprintf(stderr, "\nERROR: Could not set desired baud rate of %d Baud\n", baud);
				return false;
			}
			break;
		case 1800:
			cfsetispeed(&options, B1800);
			cfsetospeed(&options, B1800);
			break;
		case 9600:
			cfsetispeed(&options, B9600);
			cfsetospeed(&options, B9600);
			break;
		case 19200:
			cfsetispeed(&options, B19200);
			cfsetospeed(&options, B19200);
			break;
		case 38400:
			if (cfsetispeed(&options, B38400) < 0 || cfsetospeed(&options, B38400) < 0)
			{
				fprintf(stderr, "\nERROR: Could not set desired baud rate of %d Baud\n", baud);
				return false;
			}
			break;
		case 57600:
			if (cfsetispeed(&options, B57600) < 0 || cfsetospeed(&options, B57600) < 0)
			{
				fprintf(stderr, "\nERROR: Could not set desired baud rate of %d Baud\n", baud);
				return false;
			}
			break;
		case 115200:
			if (cfsetispeed(&options, B115200) < 0 || cfsetospeed(&options, B115200) < 0)
			{
				fprintf(stderr, "\nERROR: Could not set desired baud rate of %d Baud\n", baud);
				return false;
			}
			break;

		// These two non-standard (by the 70'ties ) rates are fully supported on
		// current Debian and Mac OS versions (tested since 2010).
		case 460800:
			if (cfsetispeed(&options, 460800) < 0 || cfsetospeed(&options, 460800) < 0)
			{
				fprintf(stderr, "\nERROR: Could not set desired baud rate of %d Baud\n", baud);
				return false;
			}
			break;
		case 921600:
			if (cfsetispeed(&options, 921600) < 0 || cfsetospeed(&options, 921600) < 0)
			{
				fprintf(stderr, "\nERROR: Could not set desired baud rate of %d Baud\n", baud);
				return false;
			}
			break;
		default:
			fprintf(stderr, "ERROR: Desired baud rate %d could not be set, aborting.\n", baud);
			return false;
			
			break;
	}

    // Enable the receiver and se local mode
    options.c_cflag |= (CLOCAL | CREAD); 

    // Character size, Parity & Stop bit
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    // Raw mode
    options.c_lflag  &= ~(ICANON | ECHO | ECHOE | ISIG);  /*Input*/
    options.c_oflag  &= ~OPOST;   /*Output*/

    // Disable hardware flow control
    options.c_cflag &= ~CRTSCTS;

    // Disable software flow control
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

	// Apply the configuration
	if(tcsetattr(fd, TCSAFLUSH, &options) < 0) {
		//fprintf(stderr, "\nERROR: could not set configuration of fd %d\n", fd);
		return false;
	}
	return true;
}

bool serial_write(int fd, char* buf, int len)
{
    if (write(fd, buf, len) == len)
        return true;
    else
        tcflush(fd, TCOFLUSH);
    return false;
}

bool serial_read(int fd, char* buf, int len)
{
}

void serial_close(int fd)
{
	close(fd);
}
