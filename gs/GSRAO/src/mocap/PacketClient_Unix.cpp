/*
 * Unix version of PacketClient.cpp
 * 
 * Decode NatNet packets directly.
 *
 * Written by Roice, Luo
 * 
 * Date: 2016.04.28  create this file
 */


#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ifaddrs.h>

#define MAX_NAMELENGTH              256
#define MAX_PACKETSIZE				100000	// max size of packet (actual packet size is dynamic)
#define MULTICAST_ADDRESS		"239.255.42.99"     // IANA, local network
#define PORT_DATA  			    1511

void Unpack(char*);

int main(int argc, char* argv[])
{
    struct sockaddr_in addr;
    int sock;
    int addrlen, cnt;
    struct ip_mreq mreq;
    char message[MAX_PACKETSIZE];
    
    struct ifaddrs* ifAddrStruct = NULL;
    struct ifaddrs * ifa=NULL;
    void * tmpAddrPtr=NULL;

    getifaddrs(&ifAddrStruct);

    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }
        if (ifa->ifa_addr->sa_family == AF_INET) { // check it is IP4
            // is a valid IP4 Address
            tmpAddrPtr=&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
            printf("%s IP Address %s\n", ifa->ifa_name, addressBuffer); 
        } else if (ifa->ifa_addr->sa_family == AF_INET6) { // check it is IP6
            // is a valid IP6 Address
            tmpAddrPtr=&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
            char addressBuffer[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
            printf("%s IP Address %s\n", ifa->ifa_name, addressBuffer); 
        } 
    }
    if (ifAddrStruct!=NULL) freeifaddrs(ifAddrStruct);

    /* set up socket */
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        printf("socket error");
    }

    memset((char *)&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(PORT_DATA);
    addrlen = sizeof(addr);

    /* receive */
    if (bind(sock, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        printf("bind error");
    }
    mreq.imr_multiaddr.s_addr = inet_addr(MULTICAST_ADDRESS);         
    mreq.imr_interface.s_addr = inet_addr("192.168.20.101");         
    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
		     &mreq, sizeof(mreq)) < 0) {
        printf("sssss");
    }
    while(1)
    {
        cnt = recvfrom(sock, message, sizeof(message), 0, (struct sockaddr *) &addr, (socklen_t *) &addrlen);
        if (cnt < 0) {
            printf("sssss");
        } 
        else if (cnt == 0) {
            break;
        }
        printf("%s: message = \"%s\"\n", inet_ntoa(addr.sin_addr), message);
        Unpack(message);
    }

    return 0;
}


int NatNetVersion[4] = {2,7,0,0};


bool DecodeTimecode(unsigned int inTimecode, unsigned int inTimecodeSubframe, int* hour, int* minute, int* second, int* frame, int* subframe)
{
	bool bValid = true;

	*hour = (inTimecode>>24)&255;
	*minute = (inTimecode>>16)&255;
	*second = (inTimecode>>8)&255;
	*frame = inTimecode&255;
	*subframe = inTimecodeSubframe;

	return bValid;
}

bool TimecodeStringify(unsigned int inTimecode, unsigned int inTimecodeSubframe, char *Buffer, int BufferSize)
{
	bool bValid;
	int hour, minute, second, frame, subframe;
	bValid = DecodeTimecode(inTimecode, inTimecodeSubframe, &hour, &minute, &second, &frame, &subframe);

	snprintf(Buffer,BufferSize,"%2d:%2d:%2d:%2d.%d",hour, minute, second, frame, subframe);
	for(unsigned int i=0; i<strlen(Buffer); i++)
		if(Buffer[i]==' ')
			Buffer[i]='0';

	return bValid;
}

void Unpack(char* pData)
{
    int major = NatNetVersion[0];
    int minor = NatNetVersion[1];

    char *ptr = pData;

    printf("Begin Packet\n-------\n");

    // message ID
    int MessageID = 0;
    memcpy(&MessageID, ptr, 2); ptr += 2;
    printf("Message ID : %d\n", MessageID);

    // size
    int nBytes = 0;
    memcpy(&nBytes, ptr, 2); ptr += 2;
    printf("Byte count : %d\n", nBytes);
	
    if(MessageID == 7)      // FRAME OF MOCAP DATA packet
    {
        // frame number
        int frameNumber = 0; memcpy(&frameNumber, ptr, 4); ptr += 4;
        printf("Frame # : %d\n", frameNumber);
    	
	    // number of data sets (markersets, rigidbodies, etc)
        int nMarkerSets = 0; memcpy(&nMarkerSets, ptr, 4); ptr += 4;
        printf("Marker Set Count : %d\n", nMarkerSets);

        for (int i=0; i < nMarkerSets; i++)
        {    
            // Markerset name
            char szName[256];
            strncpy(szName, ptr, 256);
            int nDataBytes = (int) strlen(szName) + 1;
            ptr += nDataBytes;
            printf("Model Name: %s\n", szName);

        	// marker data
            int nMarkers = 0; memcpy(&nMarkers, ptr, 4); ptr += 4;
            printf("Marker Count : %d\n", nMarkers);

            for(int j=0; j < nMarkers; j++)
            {
                float x = 0; memcpy(&x, ptr, 4); ptr += 4;
                float y = 0; memcpy(&y, ptr, 4); ptr += 4;
                float z = 0; memcpy(&z, ptr, 4); ptr += 4;
                printf("\tMarker %d : [x=%3.2f,y=%3.2f,z=%3.2f]\n",j,x,y,z);
            }
        }

	    // unidentified markers
        int nOtherMarkers = 0; memcpy(&nOtherMarkers, ptr, 4); ptr += 4;
        printf("Unidentified Marker Count : %d\n", nOtherMarkers);
        for(int j=0; j < nOtherMarkers; j++)
        {
            float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
            float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
            float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
            printf("\tMarker %d : pos = [%3.2f,%3.2f,%3.2f]\n",j,x,y,z);
        }
        
        // rigid bodies
        int nRigidBodies = 0;
        memcpy(&nRigidBodies, ptr, 4); ptr += 4;
        printf("Rigid Body Count : %d\n", nRigidBodies);
        for (int j=0; j < nRigidBodies; j++)
        {
            // rigid body pos/ori
            int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
            float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
            float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
            float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
            float qx = 0; memcpy(&qx, ptr, 4); ptr += 4;
            float qy = 0; memcpy(&qy, ptr, 4); ptr += 4;
            float qz = 0; memcpy(&qz, ptr, 4); ptr += 4;
            float qw = 0; memcpy(&qw, ptr, 4); ptr += 4;
            printf("ID : %d\n", ID);
            printf("pos: [%3.2f,%3.2f,%3.2f]\n", x,y,z);
            printf("ori: [%3.2f,%3.2f,%3.2f,%3.2f]\n", qx,qy,qz,qw);

            // associated marker positions
            int nRigidMarkers = 0;  memcpy(&nRigidMarkers, ptr, 4); ptr += 4;
            printf("Marker Count: %d\n", nRigidMarkers);
            int nBytes = nRigidMarkers*3*sizeof(float);
            float* markerData = (float*)malloc(nBytes);
            memcpy(markerData, ptr, nBytes);
            ptr += nBytes;
            
            if(major >= 2)
            {
                // associated marker IDs
                nBytes = nRigidMarkers*sizeof(int);
                int* markerIDs = (int*)malloc(nBytes);
                memcpy(markerIDs, ptr, nBytes);
                ptr += nBytes;

                // associated marker sizes
                nBytes = nRigidMarkers*sizeof(float);
                float* markerSizes = (float*)malloc(nBytes);
                memcpy(markerSizes, ptr, nBytes);
                ptr += nBytes;

                for(int k=0; k < nRigidMarkers; k++)
                {
                    printf("\tMarker %d: id=%d\tsize=%3.1f\tpos=[%3.2f,%3.2f,%3.2f]\n", k, markerIDs[k], markerSizes[k], markerData[k*3], markerData[k*3+1],markerData[k*3+2]);
                }

                if(markerIDs)
                    free(markerIDs);
                if(markerSizes)
                    free(markerSizes);

            }
            else
            {
                for(int k=0; k < nRigidMarkers; k++)
                {
                    printf("\tMarker %d: pos = [%3.2f,%3.2f,%3.2f]\n", k, markerData[k*3], markerData[k*3+1],markerData[k*3+2]);
                }
            }
            if(markerData)
                free(markerData);

            if(major >= 2)
            {
                // Mean marker error
                float fError = 0.0f; memcpy(&fError, ptr, 4); ptr += 4;
                printf("Mean marker error: %3.2f\n", fError);
            }

            // 2.6 and later
            if( ((major == 2)&&(minor >= 6)) || (major > 2) || (major == 0) ) 
            {
                // params
                short params = 0; memcpy(&params, ptr, 2); ptr += 2;
                bool bTrackingValid = params & 0x01; // 0x01 : rigid body was successfully tracked in this frame
            }
            
        } // next rigid body


        // skeletons (version 2.1 and later)
        if( ((major == 2)&&(minor>0)) || (major>2))
        {
            int nSkeletons = 0;
            memcpy(&nSkeletons, ptr, 4); ptr += 4;
            printf("Skeleton Count : %d\n", nSkeletons);
            for (int j=0; j < nSkeletons; j++)
            {
                // skeleton id
                int skeletonID = 0;
                memcpy(&skeletonID, ptr, 4); ptr += 4;
                // # of rigid bodies (bones) in skeleton
                int nRigidBodies = 0;
                memcpy(&nRigidBodies, ptr, 4); ptr += 4;
                printf("Rigid Body Count : %d\n", nRigidBodies);
                for (int j=0; j < nRigidBodies; j++)
                {
                    // rigid body pos/ori
                    int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
                    float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
                    float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
                    float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
                    float qx = 0; memcpy(&qx, ptr, 4); ptr += 4;
                    float qy = 0; memcpy(&qy, ptr, 4); ptr += 4;
                    float qz = 0; memcpy(&qz, ptr, 4); ptr += 4;
                    float qw = 0; memcpy(&qw, ptr, 4); ptr += 4;
                    printf("ID : %d\n", ID);
                    printf("pos: [%3.2f,%3.2f,%3.2f]\n", x,y,z);
                    printf("ori: [%3.2f,%3.2f,%3.2f,%3.2f]\n", qx,qy,qz,qw);

                    // associated marker positions
                    int nRigidMarkers = 0;  memcpy(&nRigidMarkers, ptr, 4); ptr += 4;
                    printf("Marker Count: %d\n", nRigidMarkers);
                    int nBytes = nRigidMarkers*3*sizeof(float);
                    float* markerData = (float*)malloc(nBytes);
                    memcpy(markerData, ptr, nBytes);
                    ptr += nBytes;

                    // associated marker IDs
                    nBytes = nRigidMarkers*sizeof(int);
                    int* markerIDs = (int*)malloc(nBytes);
                    memcpy(markerIDs, ptr, nBytes);
                    ptr += nBytes;

                    // associated marker sizes
                    nBytes = nRigidMarkers*sizeof(float);
                    float* markerSizes = (float*)malloc(nBytes);
                    memcpy(markerSizes, ptr, nBytes);
                    ptr += nBytes;

                    for(int k=0; k < nRigidMarkers; k++)
                    {
                        printf("\tMarker %d: id=%d\tsize=%3.1f\tpos=[%3.2f,%3.2f,%3.2f]\n", k, markerIDs[k], markerSizes[k], markerData[k*3], markerData[k*3+1],markerData[k*3+2]);
                    }

                    // Mean marker error (2.0 and later)
                    if(major >= 2)
                    {
                        float fError = 0.0f; memcpy(&fError, ptr, 4); ptr += 4;
                        printf("Mean marker error: %3.2f\n", fError);
                    }

                    // Tracking flags (2.6 and later)
                    if( ((major == 2)&&(minor >= 6)) || (major > 2) || (major == 0) ) 
                    {
                        // params
                        short params = 0; memcpy(&params, ptr, 2); ptr += 2;
                        bool bTrackingValid = params & 0x01; // 0x01 : rigid body was successfully tracked in this frame
                    }

                    // release resources
                    if(markerIDs)
                        free(markerIDs);
                    if(markerSizes)
                        free(markerSizes);
                    if(markerData)
                        free(markerData);

                } // next rigid body

            } // next skeleton
        }
        
		// labeled markers (version 2.3 and later)
		if( ((major == 2)&&(minor>=3)) || (major>2))
		{
			int nLabeledMarkers = 0;
			memcpy(&nLabeledMarkers, ptr, 4); ptr += 4;
			printf("Labeled Marker Count : %d\n", nLabeledMarkers);
			for (int j=0; j < nLabeledMarkers; j++)
			{
				// id
				int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
				// x
				float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
				// y
				float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
				// z
				float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
				// size
				float size = 0.0f; memcpy(&size, ptr, 4); ptr += 4;

                // 2.6 and later
                if( ((major == 2)&&(minor >= 6)) || (major > 2) || (major == 0) ) 
                {
                    // marker params
                    short params = 0; memcpy(&params, ptr, 2); ptr += 2;
                    bool bOccluded = params & 0x01;     // marker was not visible (occluded) in this frame
                    bool bPCSolved = params & 0x02;     // position provided by point cloud solve
                    bool bModelSolved = params & 0x04;  // position provided by model solve
                }

				printf("ID  : %d\n", ID);
				printf("pos : [%3.2f,%3.2f,%3.2f]\n", x,y,z);
				printf("size: [%3.2f]\n", size);
			}
		}

        // Force Plate data (version 2.9 and later)
        if (((major == 2) && (minor >= 9)) || (major > 2))
        {
            int nForcePlates;
            memcpy(&nForcePlates, ptr, 4); ptr += 4;
            for (int iForcePlate = 0; iForcePlate < nForcePlates; iForcePlate++)
            {
                // ID
                int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
                printf("Force Plate : %d\n", ID);

                // Channel Count
                int nChannels = 0; memcpy(&nChannels, ptr, 4); ptr += 4;

                // Channel Data
                for (int i = 0; i < nChannels; i++)
                {
                    printf(" Channel %d : ", i);
                    int nFrames = 0; memcpy(&nFrames, ptr, 4); ptr += 4;
                    for (int j = 0; j < nFrames; j++)
                    {
                        float val = 0.0f;  memcpy(&val, ptr, 4); ptr += 4;
                        printf("%3.2f   ", val);
                    }
                    printf("\n");
                }
            }
        }

		// latency
        float latency = 0.0f; memcpy(&latency, ptr, 4);	ptr += 4;
        printf("latency : %3.3f\n", latency);

		// timecode
		unsigned int timecode = 0; 	memcpy(&timecode, ptr, 4);	ptr += 4;
		unsigned int timecodeSub = 0; memcpy(&timecodeSub, ptr, 4); ptr += 4;
		char szTimecode[128] = "";
		TimecodeStringify(timecode, timecodeSub, szTimecode, 128);

        // timestamp
        double timestamp = 0.0f;
        // 2.7 and later - increased from single to double precision
        if( ((major == 2)&&(minor>=7)) || (major>2))
        {
            memcpy(&timestamp, ptr, 8); ptr += 8;
        }
        else
        {
            float fTemp = 0.0f;
            memcpy(&fTemp, ptr, 4); ptr += 4;
            timestamp = (double)fTemp;
        }

        // frame params
        short params = 0;  memcpy(&params, ptr, 2); ptr += 2;
        bool bIsRecording = params & 0x01;                  // 0x01 Motive is recording
        bool bTrackedModelsChanged = params & 0x02;         // 0x02 Actively tracked model list has changed


		// end of data tag
        int eod = 0; memcpy(&eod, ptr, 4); ptr += 4;
        printf("End Packet\n-------------\n");

    }
    else if(MessageID == 5) // Data Descriptions
    {
        // number of datasets
        int nDatasets = 0; memcpy(&nDatasets, ptr, 4); ptr += 4;
        printf("Dataset Count : %d\n", nDatasets);

        for(int i=0; i < nDatasets; i++)
        {
            printf("Dataset %d\n", i);

            int type = 0; memcpy(&type, ptr, 4); ptr += 4;
            printf("Type : %d\n", i, type);

            if(type == 0)   // markerset
            {
                // name
                char szName[256];
                strncpy(szName, ptr, 256);
                int nDataBytes = (int) strlen(szName) + 1;
                ptr += nDataBytes;
                printf("Markerset Name: %s\n", szName);

        	    // marker data
                int nMarkers = 0; memcpy(&nMarkers, ptr, 4); ptr += 4;
                printf("Marker Count : %d\n", nMarkers);

                for(int j=0; j < nMarkers; j++)
                {
                    char szName[256];
                    strncpy(szName, ptr, 256);
                    int nDataBytes = (int) strlen(szName) + 1;
                    ptr += nDataBytes;
                    printf("Marker Name: %s\n", szName);
                }
            }
            else if(type ==1)   // rigid body
            {
                if(major >= 2)
                {
                    // name
                    char szName[MAX_NAMELENGTH];
                    strncpy(szName, ptr, 256);
                    ptr += strlen(ptr) + 1;
                    printf("Name: %s\n", szName);
                }

                int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
                printf("ID : %d\n", ID);
             
                int parentID = 0; memcpy(&parentID, ptr, 4); ptr +=4;
                printf("Parent ID : %d\n", parentID);
                
                float xoffset = 0; memcpy(&xoffset, ptr, 4); ptr +=4;
                printf("X Offset : %3.2f\n", xoffset);

                float yoffset = 0; memcpy(&yoffset, ptr, 4); ptr +=4;
                printf("Y Offset : %3.2f\n", yoffset);

                float zoffset = 0; memcpy(&zoffset, ptr, 4); ptr +=4;
                printf("Z Offset : %3.2f\n", zoffset);

            }
            else if(type ==2)   // skeleton
            {
                char szName[MAX_NAMELENGTH];
                strncpy(szName, ptr, 256);
                ptr += strlen(ptr) + 1;
                printf("Name: %s\n", szName);

                int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
                printf("ID : %d\n", ID);

                int nRigidBodies = 0; memcpy(&nRigidBodies, ptr, 4); ptr +=4;
                printf("RigidBody (Bone) Count : %d\n", nRigidBodies);

                for(int i=0; i< nRigidBodies; i++)
                {
                    if(major >= 2)
                    {
                        // RB name
                        char szName[MAX_NAMELENGTH];
                        strncpy(szName, ptr, 256);
                        ptr += strlen(ptr) + 1;
                        printf("Rigid Body Name: %s\n", szName);
                    }

                    int ID = 0; memcpy(&ID, ptr, 4); ptr +=4;
                    printf("RigidBody ID : %d\n", ID);

                    int parentID = 0; memcpy(&parentID, ptr, 4); ptr +=4;
                    printf("Parent ID : %d\n", parentID);

                    float xoffset = 0; memcpy(&xoffset, ptr, 4); ptr +=4;
                    printf("X Offset : %3.2f\n", xoffset);

                    float yoffset = 0; memcpy(&yoffset, ptr, 4); ptr +=4;
                    printf("Y Offset : %3.2f\n", yoffset);

                    float zoffset = 0; memcpy(&zoffset, ptr, 4); ptr +=4;
                    printf("Z Offset : %3.2f\n", zoffset);
                }
            }

        }   // next dataset

       printf("End Packet\n-------------\n");

    }
    else
    {
        printf("Unrecognized Packet Type.\n");
    }

}
