/*
 * Unix version of PacketClient.cpp
 * 
 * Decode NatNet packets directly.
 *
 * Written by Roice, Luo
 * 
 * Date: 2016.05.20  create this file
 */
#ifndef PACKET_CLIENT_H
#define PACKET_CLIENT_H

#include <string>

typedef struct {
    int rbID; // rigid body ID
    float pos[3]; // x,y,z
    float ori[4]; // qx,qy,qz,qw
    float enu[3]; // east, north, up
    float att[3]; // roll, pitch, yaw
} MocapRigidBody_t;

typedef struct {
    int MessageID;
    int nBytes;
    int frameNumber;
    int nMarkerSets;
    MocapRigidBody_t robot[4]; // 4 robots max
    float latency;
    float dlatency;
} MocapData_t;

/* robot[i] contains "Rigid Body X" */
typedef struct {
    int rigid_body_id[4]; // 4 robots max
} MocapReq_t; // mocap request

bool mocap_client_init(const char*);
void mocap_client_close(void);
MocapData_t* mocap_get_data(void);
void mocap_set_request(std::string*);

#endif
