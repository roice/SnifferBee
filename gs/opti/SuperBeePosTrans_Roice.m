                                                  
% Optitrack Matlab 
%3-25 串口发数不能使用异步方式

% serial callback
function instrcallback(obj, event)
    global ComQuerySize;
    [out, a] = fread(obj, ComQuerySize, 'uint8');
end
    

function NatNetMatlabSample()

    display('NatNet Sample Begin')
    global ini;
    global frameRate;
    global y_d;
    global tsec;
    global lock;
    global count;
    global gusture_roll_angle_sum;
    global controlled_by_indicator;
    global gusture_roll_angle_last;
    gusture_roll_angle_sum=0;
    gusture_roll_angle_last=0;
    controlled_by_indicator=-1;
    lock=0;
    count=0;
    tsec=0;
    y_d = -1;
    global x_d;
    global z_d;
    x_d = 0;   %0 
    z_d = 0;  %0
    ini=1;
    lastFrameTime = -1.0;
    lastFrameID = -1.0;
    usePollingLoop = false;         % approach 1 : poll for mocap data in a tight loop using GetLastFrameOfData
    usePollingTimer = true;        % approach 2 : poll using a Matlab timer callback ( better for UI based apps )
    useFrameReadyEvent = false;      % 3-25测试定时器控制approach 3 : use event callback from NatNet (no polling)
    useUI = true;
    %serial port
%     display('AAAAA')
    delete(instrfindall);%释放所有串口 
    global sscom;
    global ComQuerySize = 4;
    sscom=serial('com16');
    set(sscom,'BaudRate',57600);
    sscom.Timeout = 30;
    sscom.ReadAsyncMode = 'continuous';
    sscom.BytesAvailableFcnMode = 'byte';
    sscom.BytesAvailableFcnCount = ComQuerySize;
    sscom.BytesAvailableFcn = @instrcallback;
    fopen(sscom);
    %fwrite(sscom,fix(2020/256),'char','async');
    global x_file_save;
    global y_file_save;
    global z_file_save;
    global r_file_save;
    global p_file_save;
    global yaw_file_save;
    x_file_save = fopen('xData.txt', 'w+');  
    y_file_save = fopen('yData.txt', 'w+');
    z_file_save = fopen('zData.txt', 'w+');
    r_file_save = fopen('rData.txt', 'w+');
    p_file_save = fopen('pData.txt', 'w+');
    yaw_file_save = fopen('yawData.txt', 'w+');    
    persistent arr; 
    % Open figure
    if(useUI)
        hFigure = figure('Name','OptiTrack NatNet Matlab Sample','NumberTitle','off');
    end

    try
        % Add NatNet .NET assembly so that Matlab can access its methods, delegates, etc.
        % Note : The NatNetML.DLL assembly depends on NatNet.dll, so make sure they
        % are both in the same folder and/or path if you move them.
        display('[NatNet] Creating Client.')
		curDir = pwd;
		mainDir = fileparts(fileparts(curDir));
		dllPath = fullfile(mainDir,'lib','x64','NatNetML.dll');
        assemblyInfo = NET.addAssembly(dllPath);

        % Create an instance of a NatNet client
        theClient = NatNetML.NatNetClientML(0); % Input = iConnectionType: 0 = Multicast, 1 = Unicast
        version = theClient.NatNetVersion();
        fprintf( '[NatNet] Client Version : %d.%d.%d.%d\n', version(1), version(2), version(3), version(4) );

        % Connect to an OptiTrack server (e.g. Motive)
        display('[NatNet] Connecting to OptiTrack Server.')
        hst = java.net.InetAddress.getLocalHost;
        %HostIP = char(hst.getHostAddress);
        %HostIP = char('239.255.42.99');
        HostIP = char('127.0.0.1');
        flg = theClient.Initialize(HostIP, HostIP); % Flg = returnCode: 0 = Success
        if (flg == 0)
            display('[NatNet] Initialization Succeeded')
        else
            display('[NatNet] Initialization Failed')
        end
        
        % print out a list of the active tracking Models in Motive
        GetDataDescriptions(theClient)
        
        % Test - send command/request to Motive
        [byteArray, retCode] = theClient.SendMessageAndWait('FrameRate');
        if(retCode ==0)
            byteArray = uint8(byteArray);
            frameRate = typecast(byteArray,'single');
        end

        % get the mocap data
        if(usePollingTimer)
            % approach 2 : poll using a Matlab timer callback ( better for UI based apps )
            framePerSecond = 100;   % 200HZ to 50HZ for remote control timer frequency
            TimerData = timer('TimerFcn', {@TimerCallback,theClient},'Period',1/framePerSecond,'ExecutionMode','fixedRate','BusyMode','drop');
            start(TimerData);
            % wait until figure is closed
            uiwait(hFigure);
        else
            if(usePollingLoop)
                % approach 1 : get data by polling - just grab 5 secs worth of data in a tight loop
                for idx = 1 : 1000   
                   % Note: sleep() accepts [mSecs] duration, but does not process any events.
                   % pause() processes events, but resolution on windows can be at worst 15 msecs
                   java.lang.Thread.sleep(5);  

                    % Poll for latest frame instead of using event callback
                    data = theClient.GetLastFrameOfData();
                    frameTime = data.fLatency;
                    frameID = data.iFrame;
                    if(frameTime ~= lastFrameTime)
                        fprintf('FrameTime: %0.3f\tFrameID: %5d\n',frameTime, frameID);
                        lastFrameTime = frameTime;
                        lastFrameID = frameID;
                    else
                        display('Duplicate frame');
                    end
                 end
            else
                % approach 3 : get data by event handler (no polling)
                % Add NatNet FrameReady event handler
                ls = addlistener(theClient,'OnFrameReady2',@(src,event)FrameReadyCallback(src,event));
                display('[NatNet] FrameReady Listener added.');
                % wait until figure is closed
                uiwait(hFigure);
            end
        end

    catch err
        display(err);
    end

    % cleanup
    if(usePollingTimer)
        stop(TimerData);
        delete(TimerData);
    end
    theClient.Uninitialize();
    if(useFrameReadyEvent)
        if(~isempty(ls))
            delete(ls);
        end
    end
    clear functions;

    display('NatNet Sample End')
    
end

% Test : process data in a Matlab Timer callback
function TimerCallback(obj, event, theClient)
    frameOfData = theClient.GetLastFrameOfData();
    UpdateUI( frameOfData );
    %display('callback')
   % fwrite(sscom,fix(2020/256),'char','async');
    
end

% Test : Process data in a NatNet FrameReady Event listener callback
function FrameReadyCallback(src, event)
    
    frameOfData = event.data;
    UpdateUI( frameOfData );
    
end

% Update a Matlab Plot with values from a single frame of mocap data
function UpdateUI( frameOfData )

    persistent lastFrameTime;
    persistent lastFrameID;
    persistent hX;
    persistent hY;
    persistent hZ;
    persistent RX;
    persistent PY;
    persistent YZ;    
    persistent arrayIndex;
    persistent frameVals;
    persistent xVals;
    persistent yVals;
    persistent zVals;
    persistent bufferModulo;
    persistent xPos;
    persistent yPos;
    persistent zPos;
    global frameRate;
    global sscom;
    global x_file_save;
    global y_file_save;
    global z_file_save;
    global r_file_save;
    global p_file_save;
    global yaw_file_save;   
    global y_d;
    global lock;
    global x_d;
    global z_d;
    global count;
    global gusture_roll_angle_sum;
    global controlled_by_indicator;
    global gusture_roll_angle_last;
    global go_round;
    global square_start_poX;
    global square_start_poZ;
    global go_spiral;
    global go_square;
    global spiral_bottom_pos;
    global circle_centerX;
    global circle_centerZ;
    global init;
    persistent time_period;
    persistent pos_x_NEE;
    persistent pos_y_NEE;
    persistent pos_z_NEE;
    persistent last_pos_x_NEE;
    persistent last_pos_y_NEE;
    persistent last_pos_z_NEE;
    persistent roll;
    persistent pitch;
    global yaw;
   
    global    positionX;
    global    positionY;
    global    positionZ;
    global    tsec;
    % first time - generate an array and a plot
    if isempty(hX)
        % initialize statics
        bufferModulo = 256;
        frameVals = 1:255;
        xVals = zeros([1,255]);
        yVals = zeros([1,255]);
        zVals = zeros([1,255]);
        xPos= zeros([1,255]);
        yPos = zeros([1,255]);
        zPos = zeros([1,255]);
        arrayIndex = 1;
        lastFrameTime = frameOfData.fLatency;
        lastFrameID = frameOfData.iFrame;
       
        % create plot
       % hX = plot(frameVals, xVals, 'color', 'r');
        figure(1);
        subplot(211)
        hold on;
        %hY = plot(frameVals, yVals, 'color', 'r');
        %hZ = plot(frameVals, zVals, 'color', 'r');
        hX = plot(frameVals, xPos, 'color', 'r');
        hY = plot(frameVals, yPos, 'color', 'g');
        hZ = plot(frameVals, zPos, 'color', 'b');
        title('Mocap Position Plot');
        xlabel('Frame number');
        ylabel('Position(m)');
        %set(gca,'YLim',[-1.5 1.5]);    
        set(gca,'XGrid','on','YGrid','on');
        subplot(212)
         hold on;
        RX = plot(frameVals, xVals, 'color', 'r');
        PY = plot(frameVals, yVals, 'color', 'g');
        YZ = plot(frameVals, zVals, 'color', 'b');
        title('Mocap Angle Plot');
        xlabel('Frame number');
        ylabel('Angle(degree)');
        %set(gca,'YLim',[-1.5 1.5]);    
        set(gca,'XGrid','on','YGrid','on');
        
    end

    % calculate the frame increment based on mocap frame's timestamp
    % in general this should be monotonically increasing according
    % To the mocap framerate, however frames are not guaranteed delivery
    % so to be accurate we test and report frame drop or duplication
    newFrame = true;
    droppedFrames = false;
    frameTime = frameOfData.fLatency;
    frameID = frameOfData.iFrame;
    calcFrameInc = round( (frameTime - lastFrameTime) * frameRate );
    % clamp it to a circular buffer of 255 frames
    arrayIndex = mod(arrayIndex + calcFrameInc, bufferModulo);
    if(arrayIndex==0)
        arrayIndex = 1;
    end
    if(calcFrameInc > 1)
        % debug
        % fprintf('\nDropped Frame(s) : %d\n\tLastTime : %.3f\n\tThisTime : %.3f\n', calcFrameInc-1, lastFrameTime, frameTime);
        droppedFrames = true;
    elseif(calcFrameInc == 0)
        % debug
        % display('Duplicate Frame')      
        newFrame = false;
    end
    
    % debug
     %fprintf('FrameTime: %0.3f\tFrameID: %d\n',frameTime, frameID);
    %fprintf('FrameRate: %0.3f\tFrameID: %d\n',frameRate, frameID);
%    中间值
    r_command = 1520; 
    p_command = 17904;
    t_command = 34288;
    y_command = 50672;
%    最大值
%     r_command = 2020;
%     p_command = 18404;
%     t_command = 34788;
%     y_command = 51172;
%     r_command = 1520;
%     p_command = 17904;
%     t_command = 34288; %+ int8(zPos);
%     y_command = 50672;
%     %fprintf('\n',xPos, frameID);
%     fwrite(sscom,fix(r_command/256),'char');%先发高字节
%     fwrite(sscom,mod(r_command,256),'char');%后发低字节
%     fwrite(sscom,fix(p_command/256),'char');%先发高字节
%     fwrite(sscom,mod(p_command,256),'char');%后发低字节
%     fwrite(sscom,fix(t_command/256),'char');%先发高字节
%     fwrite(sscom,mod(t_command,256),'char');%后发低字节
%     fwrite(sscom,fix(y_command/256),'char');%先发高字节
%     fwrite(sscom,mod(y_command,256),'char');%后发低字节
    try
        if(newFrame)
            if(frameOfData.RigidBodies.Length() > 0)

                rigidBodyData = frameOfData.RigidBodies(1);
                rigidBodyData2 =frameOfData.RigidBodies(2);
                %xPos = data.LabeledMarkers(1).x;
                %yPos = data.LabeledMarkers(1).y;
                %zPos = data.LabeledMarkers(1).z;
                % Test : Marker Y Position Data
                 %angleY = data.LabeledMarkers(1).y;

                % Test : Rigid Body Y Position Data
%                 xPos = rigidBodyData.x;
%                 yPos = rigidBodyData.y;
%                 zPos = rigidBodyData.z;
                %fprintf('\n',xPos, frameID);
%                 xPos
%                 yPos
%                 zPos
                % Test : Rigid Body 'Yaw'
                % Note : Motive display euler's is X (Pitch), Y (Yaw), Z (Roll), Right-Handed (RHS), Relative Axes
                % so we decode eulers heres to match that.
                q = quaternion( rigidBodyData.qx, rigidBodyData.qy, rigidBodyData.qz, rigidBodyData.qw );
                qRot = quaternion( 0, 0, 0, 1);     % rotate pitch 180 to avoid 180/-180 flip for nicer graphing
                q = mtimes(q, qRot);
                angles = EulerAngles(q,'zyx');
                angleX = -angles(1) * 180.0 / pi;   % must invert due to 180 flip above
                angleY = angles(2) * 180.0 / pi;
                angleZ = -angles(3) * 180.0 / pi;   % must invert due to 180 flip above
                %%%%-------------
                 q2 = quaternion( rigidBodyData2.qx, rigidBodyData2.qy, rigidBodyData2.qz, rigidBodyData2.qw );
                qRot2 = quaternion( 0, 0, 0, 1);     % rotate pitch 180 to avoid 180/-180 flip for nicer graphing
                q2 = mtimes(q2, qRot2);
                angles2 = EulerAngles(q2,'zyx');
                angleX2 = -angles2(1) * 180.0 / pi;   % must invert due to 180 flip above
                angleY2 = angles2(2) * 180.0 / pi;
                angleZ2 = -angles2(3) * 180.0 / pi;   % must invert due to 180 flip above
                
                
                
                
                %%%%--------
                if(droppedFrames)
                    for i = 1 : calcFrameInc
                        fillIndex = arrayIndex - i;
                        if(fillIndex < 1)
                            fillIndex = bufferModulo-(abs(fillIndex)+1);
                        end
                        xVals(fillIndex) = angleX;  
                        yVals(fillIndex) = angleY;  
                        zVals(fillIndex) = angleZ;  
                        xPos(fillIndex) = rigidBodyData.x;  
                        yPos(fillIndex) = rigidBodyData.y;  
                        zPos(fillIndex) = rigidBodyData.z;
                    end
                end

                % update the array/plot for this frame
                xVals(arrayIndex) = angleX;  
                yVals(arrayIndex) = angleY;  
                zVals(arrayIndex) = angleZ;  
                xPos(arrayIndex) = rigidBodyData.x;  
                yPos(arrayIndex) = rigidBodyData.y;  
                zPos(arrayIndex) = rigidBodyData.z;
                set(hX, 'YData', xPos);
                set(hY, 'YData', yPos);
                set(hZ, 'YData', zPos);
                set(RX, 'YData', xVals);
                set(PY, 'YData', yVals);
                set(YZ, 'YData', zVals);
                set(gcf,'keypressfcn',@keytest);%刷新按键
      %中间值
    r_command = 1520; 
    p_command = 17904;
    t_command = 34288;
    y_command = 50672;

pos_x_NEE= -rigidBodyData.z*100;
pos_y_NEE= rigidBodyData.x*100;
pos_z_NEE= -rigidBodyData.y*100;
positionX=rigidBodyData.x;
positionY=rigidBodyData.y;
positionZ=rigidBodyData.z;
init=0;
if rigidBodyData.y>0.2
init=1;
end  
if init==0   % initialize parameters to avoid error
circle_centerX=0;
circle_centerZ=0;
go_spiral=-1;
go_square=-1;
spiral_vel_y=5;
square_vel_x=15;
square_vel_z=0;
square_start_poX=0;
square_start_poZ=0;
spiral_bottom_pos=0;
go_round=-1;
find_max=0;
last_pos_x_NEE=pos_x_NEE;
last_pos_y_NEE=pos_y_NEE;
last_pos_z_NEE=pos_z_NEE;
end
time_period=1/50;
vel_x_NEE=(pos_x_NEE-last_pos_x_NEE)/time_period;
vel_y_NEE=(pos_y_NEE-last_pos_y_NEE)/time_period;
vel_z_NEE=(pos_z_NEE-last_pos_z_NEE)/time_period;
last_pos_x_NEE=pos_x_NEE;
last_pos_y_NEE=pos_y_NEE;
last_pos_z_NEE=pos_z_NEE;

%----------------------------------------------
desire_pos_x_NEE= -z_d*100;
desire_pos_y_NEE= x_d*100;
desire_pos_z_NEE= -y_d*100;
Cr=cos(pi*angleZ/180); 
Sr=sin(pi*angleZ/180);
Cp=cos(pi*angleX/180);
Sp=sin(pi*angleX/180);
Cy=cos(pi*angleY/180);
Sy=sin(pi*angleY/180);
R=[-Cy*Cp Sp*Sr-Sy*Cp*Cr Sy*Cp*Sr+Cr*Sp;Sy -Cy*Cr Cy*Sr;Cy*Sp Cp*Sr+Sy*Cr*Sp Cp*Cr-Sy*Sp*Sr];
pitch=asin(-R(3,1));
sinyaw=R(2,1)/cos(pitch);
cosyaw=R(1,1)/cos(pitch);
sinroll=R(3,2)/cos(pitch);
cosroll=R(3,3)/cos(pitch);
yaw=2*pi-acos(cosyaw);
if sinyaw>0 & cosyaw>0
  yaw=asin(sinyaw);
end
if  sinyaw>0 & cosyaw<0
  yaw=acos(cosyaw);    
end

roll=asin(sinroll);
if sinroll>0 & cosroll<0
  roll=acos(cosroll);
end
if sinroll<0 & cosroll<0
   roll=-acos(cosroll);
end
attitude=[roll*180/pi pitch*180/pi yaw*180/pi];
% generate APM DCM matrix
SSy=sin(yaw);
CCy=cos(yaw);
SSr=sin(roll);
CCr=cos(roll);
SSp=sin(pitch);
CCp=cos(pitch);
%---------------rigid body2
Cr2=cos(pi*angleZ2/180); 
Sr2=sin(pi*angleZ2/180);
Cp2=cos(pi*angleX2/180);
Sp2=sin(pi*angleX2/180);
Cy2=cos(pi*angleY2/180);
Sy2=sin(pi*angleY2/180);
DcmB2O=[Cr2*Cy2 -Cr2*Sr2 Sy2;Cp2*Sr2+Cr2*Sp2*Sy2 Cp2*Cr2-Sp2*Sr2*Sy2 -Cy2*Sp2;Sp2*Sr2-Cp2*Cr2*Sy2 Cr2*Sp2+Cp2*Sr2*Sy2 Cp2*Cy2];
posIndicator=[rigidBodyData2.x*100 ; rigidBodyData2.y*100 ; rigidBodyData2.z*100];
att=[angleZ2 angleX2 angleY2];
posTarget=posIndicator+DcmB2O*[0;0;150];
add_angle=angleZ2-gusture_roll_angle_last;
if add_angle<-100
   add_angle=0; 
end
if add_angle>100
   add_angle=0; 
end
gusture_roll_angle_sum=gusture_roll_angle_sum+add_angle;
gusture_roll_angle_sum=gusture_roll_angle_sum-20;
if gusture_roll_angle_sum<0
    gusture_roll_angle_sum=0;
    lock=0;
end
gusture_roll_angle_last=angleZ2;
[gusture_roll_angle_sum controlled_by_indicator]
if gusture_roll_angle_sum>50
   if lock==0
   controlled_by_indicator=-controlled_by_indicator
   lock=1;
   end
end
posTarget=[0 0 -1;1 0 0;0 -1 0]*posTarget;
% To ensure that quad is in the safe area
posTarget(1,1)=constrain(posTarget(1,1),-200,200);
posTarget(2,1)=constrain(posTarget(2,1),-200,200);
posTarget(3,1)=constrain(-posTarget(3,1),0,150);
persistent countprint;
    init=1;
     count=count+1;
   
    tsec=tsec+0.02;
    parity=0;
    %transdata=['$OPT,' num2str(tsec) ',' num2str(pos_x_NEE*100) ',' num2str(pos_y_NEE*100) ',' num2str(-pos_z_NEE*100) ',' num2str(roll*18000/pi) ',' num2str(pitch*18000/pi) ',' num2str(yaw*18000/pi) ',,*' ]
   %transdata=['$OPT,' num2str(tsec) ',' num2str(floor(pos_x_NEE*100)) ',' num2str(floor(pos_y_NEE*100)) ',' num2str(-floor(pos_z_NEE*100)) ',' num2str(floor(roll*18000/pi)) ',' num2str(floor(pitch*18000/pi)) ',' num2str(floor(yaw*18000/pi)) ',,*' ] 
   %transdata=['$OPT,' num2str(tsec) ',' num2str(floor(pos_x_NEE*100)) ',' num2str(floor(pos_y_NEE*100)) ',' num2str(-floor(pos_z_NEE*100)) ',' num2str(floor(posTarget(1,1)*100)) ',' num2str(floor(posTarget(2,1)*100)) ',' num2str(floor(posTarget(3,1)*100)) ',' num2str(controlled_by_indicator) ',,*' ];
  
% For Debug
  %pos_x_NEE
  %pos_y_NEE
  %pos_z_NEE
  pos_ENU_e = int32(pos_y_NEE*100)
  pos_ENU_n = int32(pos_x_NEE*100)
  pos_ENU_u = int32(pos_z_NEE*(-100))
  
   % preparing SuperBee SBSP message
   % message name: SBSP_FRESH_POS_OPT
   % message id: 0x3d
   sbsp_preamble = '$B';
   sbsp_direction = '<';
   sbsp_size = uint8(12);
   sbsp_cmd = uint8(61);
   sbsp_data = [pos_ENU_e, pos_ENU_n, pos_ENU_u];
   
   % generate data sequence
   trans_seq(1) = uint8('$');
   trans_seq(2) = uint8('B');
   trans_seq(3) = uint8('<');
   trans_seq(4) = sbsp_size;
   trans_seq(5) = sbsp_cmd;
   trans_seq = [trans_seq, typecast(sbsp_data, 'uint8')];
   
   % calculate CRC
   sbsp_crc = trans_seq(4);
   for ii=5:1:(size(trans_seq,2))
        sbsp_crc=bitxor(uint8(sbsp_crc),uint8(trans_seq(ii)));
   end
   trans_seq(size(trans_seq,2)+1) = sbsp_crc;
   %-- End of data stream generation
 
   % transmit to serial port
   fwrite(sscom, trans_seq);
     
   %  parity=dec2hex(o);
  %   fwrite(sscom,parity);
%       fwrite(sscom,parity);
%      fwrite(sscom,'$GPGGA,');
%      fwrite(sscom,num2str(tsec)); =-=-----
%      a=num2str(tsec)
%      fwrite(sscom,',');
%      fwrite(sscom,num2str(pos_x_NEE)); 
%      fwrite(sscom,',');
%      fwrite(sscom,num2str(pos_y_NEE)); 
%      fwrite(sscom,',');
%      fwrite(sscom,num2str(pos_z_NEE));
%      fwrite(sscom,',');
%      fwrite(sscom,num2str(roll*180/pi));
%      fwrite(sscom,',');
%      fwrite(sscom,num2str(pitch*180/pi));
%      fwrite(sscom,',');
%      fwrite(sscom,num2str(yaw*180/pi));
%      fwrite(sscom,',,*');
%      parity=0;
%      fwrite(sscom,num2str(parity));
   %  fwrite(sscom,49,'uint8');%先发高字节
  %   fwrite(sscom,mod(p_command,256),'char');%后发低字节
%     fwrite(sscom,fix(uint16(t_command+temp-500)/256)-1,'char');%先发高字节
%     fwrite(sscom,mod(uint16(t_command+temp-500),256),'char');%后发低字节
%     fwrite(sscom,fix(y_command/256),'char');%先发高字节
%     fwrite(sscom,mod(y_command,256),'char');%后发低字节
%     fwrite(sscom,uint16(r_command+temp_roll-500),'uint16');
%     fwrite(sscom,uint16(p_command+temp_pitch-500),'uint16');
%     fwrite(sscom,uint16(t_command+temp_t-500),'uint16');
%     fwrite(sscom,uint16(y_command+temp_yaw-500),'uint16');
    fprintf(x_file_save, '%f ', rigidBodyData.x);  
    fprintf(y_file_save, '%f ', rigidBodyData.y);
    fprintf(z_file_save, '%f ', rigidBodyData.z);
    fprintf(r_file_save, '%f ', angleZ);  
     fprintf(p_file_save, '%f ', angleX);
     fprintf(yaw_file_save, '%f ', angleY);    
 %    fprintf(r_file_save, '%f ', x_d*100);  
%     fprintf(p_file_save, '%f ', y_d*100);
 %    fprintf(yaw_file_save, '%f ', z_d*100); 
           end
        end
    catch err
       % display(err);
    end
    
    lastFrameTime = frameTime;
    lastFrameID = frameID;

end

% Print out a description of actively tracked models from Motive
function GetDataDescriptions( theClient )

    dataDescriptions = theClient.GetDataDescriptions();
    
    % print out 
    fprintf('[NatNet] Tracking Models : %d\n\n', dataDescriptions.Count);
    for idx = 1 : dataDescriptions.Count
        descriptor = dataDescriptions.Item(idx-1);
        if(descriptor.type == 0)
            fprintf('\tMarkerSet \t: ');
        elseif(descriptor.type == 1)
            fprintf('\tRigid Body \t: ');                
        elseif(descriptor.type == 2)
            fprintf('\tSkeleton \t: ');               
        else
            fprintf('\tUnknown data type : ');               
        end
        fprintf('%s\n', char(descriptor.Name));
    end

    for idx = 1 : dataDescriptions.Count
        descriptor = dataDescriptions.Item(idx-1);
        if(descriptor.type == 0)
            fprintf('\n\tMarkerset : %s\t(%d markers)\n', char(descriptor.Name), descriptor.nMarkers);
            markerNames = descriptor.MarkerNames;
            for markerIndex = 1 : descriptor.nMarkers
                name = markerNames(markerIndex);
                fprintf('\t\tMarker : %-20s\t(ID=%d)\n', char(name), markerIndex);             
            end
        elseif(descriptor.type == 1)
            fprintf('\n\tRigid Body : %s\t\t(ID=%d, ParentID=%d)\n', char(descriptor.Name),descriptor.ID,descriptor.parentID);
        elseif(descriptor.type == 2)
            fprintf('\n\tSkeleton : %s\t(%d bones)\n', char(descriptor.Name), descriptor.nRigidBodies);
            %fprintf('\t\tID : %d\n', descriptor.ID);
            rigidBodies = descriptor.RigidBodies;
            for boneIndex = 1 : descriptor.nRigidBodies
                rigidBody = rigidBodies(boneIndex);
                fprintf('\t\tBone : %-20s\t(ID=%d, ParentID=%d)\n', char(rigidBody.Name), rigidBody.ID, rigidBody.parentID);
            end               
        end
    end

end
% 按键响应函数
function keytest(h,e)
global y_d;
global x_d;
global z_d;
global positionX;
global positionY;
global positionZ;
global desire_yaw;
global rotate_yaw;
global go_round;
global go_square;
global square_start_poX;
global square_start_poZ;
global go_spiral;
global spiral_bottom_pos;
global circle_centerX;
global circle_centerZ;
disp('pressed');
key = get(gcf,'currentcharacter');
if key == 'l'
    display('down!down!');
    y_d = y_d - 0.1;
end
if key =='u'
    display('up!up!')
    y_d = y_d + 0.1;   
end
if key == 'a'
    display('left!left!');
    x_d = x_d + 0.1;
end
if key =='d'
    display('right!right!')
    x_d = x_d - 0.1;
end
if key == 'w'
    display('forth!forth!');
    z_d = z_d + 0.1;
end
if key =='s'
    display('back!back!')
    z_d = z_d - 0.1;
end
if key=='t'
    y_d=0.6;
end
if key== '6'
    desire_yaw=desire_yaw+300;
    if desire_yaw>=36000
        desire_yaw=desire_yaw-36000;
    end
end
if key== '4'
    desire_yaw=desire_yaw-300;
    if desire_yaw<0
        desire_yaw=desire_yaw+36000;
    end     
end
if key== '5'
    desire_yaw=18000;
end

if key=='r'
    rotate_yaw=-rotate_yaw;
end
if key=='o'
    go_round=-go_round;
    circle_centerX=positionX-1;
    circle_centerZ=positionZ;
end
if key=='0'
    go_spiral=-go_spiral;
    go_round=-go_round;
    spiral_bottom_pos=positionY;
    circle_centerX=positionX-1;
    circle_centerZ=positionZ;
end
if key=='q'
    go_square=-go_square;
    square_start_poX=positionX;
    square_start_poZ=positionZ;
end
end
function retValX=pid_rate_X(err_velX,kp,ki,kd,imax,dt)
persistent ix;
persistent dx;
persistent last_err_velX;
persistent last_dx;
global init;
if init==0
   ix=0;
   last_err_velX=0;
   last_dx=0;
end
ix=ix+(err_velX - last_err_velX)*dt*ki;
if ix>imax
   ix=imax;
end  
if ix<-imax
   ix=-imax;
end
px=kp*err_velX; 
dx = kd*(err_velX - last_err_velX) / dt;
dx = last_dx + (0.02 / ( 0.00398089 + 0.02)) * (dx - last_dx);
last_err_velX  = err_velX;
last_dx = dx;
retValX= px+ix+dx;
end

function retValY=pid_rate_Y(err_velY,kp,ki,kd,imax,dt)
persistent py;
persistent iy;
persistent dy;
persistent last_err_velY;
persistent last_dy;
global init;
if init==0
   iy=0;
   last_err_velY=0; 
   last_dy=0;
end
iy=iy+(err_velY - last_err_velY)*dt*ki;
if iy>imax
   iy=imax;
end  
if iy<-imax
   iy=-imax;
end
py=kp*err_velY;  
dy = kd*(err_velY - last_err_velY) / dt;
dy = last_dy + (0.02 / ( 0.00398089 + 0.02)) * (dy - last_dy);
last_err_velY  = err_velY;
last_dy = dy;
retValY= py+iy+dy;
end
function ret_angle=wrap_180(angle)
if angle >18000
    angle=angle-36000;
end
if angle<-18000
    angle=angle+36000;
end
ret_angle=angle;
end
function ret_angle_cs=constrain(angle,low,high)
if angle<low
    angle=low;
end
if angle>high
    angle=high;
end
ret_angle_cs=angle;
end
 
