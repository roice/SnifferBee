function [] = plot_motion(filename)

thedata = load(filename);

A = size(thedata)
attributes = A(2)
entries = A(1)


hold on
%overlay the dog's path
%plot(thedata(1:entries,12),thedata(1:entries,13))

Latgpsmu = []; %create vectors to store this data
Longgpsmu = [];

%filter out the really crazy stuff

for i=1:entries
    if ( thedata(i,8) > -80 | thedata(i,8) < -90 | thedata(i,9) > 40 | thedata(i,9) < 30)
        'rejected'
        i
    else
        Latgpsmu = [Latgpsmu thedata(i,9)];
        Longgpsmu = [Longgpsmu thedata(i,8)];
    end
                
end

time = 0:(94/entries):entries

size(time)

forward = [];
stop = [];
left = [];
right = [];
recall = [];

dec = 3.294;

for i = 1:entries
    if (thedata(i,6) < 0)
        thedata(i,6) = thedata(i,6)+360-dec;
    else
        thedata(i,6) = thedata(i,6)-dec;
    end
    
    %forwards
    if (thedata(i,23) == 49)
        forward = [forward 1];
    else
        forward = [forward 0];
    end
    
    
        %stops
    if (thedata(i,23) == 51)
        stop = [stop 1];
    else
        stop = [stop 0];
    end
    
    %lefts
    if (thedata(i,23) == 52)
        left = [left 1];
    else
        left = [left 0];
    end
    
    %rights
    if (thedata(i,23) == 55)
        right = [right 1];
    else
        right = [right 0];
    end
    
    
        %recalls
    if (thedata(i,23) == 57)
        recall = [recall 1];
    else
        recall = [recall 0];
    end
end

outliersforward = excludedata(1:entries,forward,'range',[0 0.5]);
outliersstop = excludedata(1:entries,stop,'range',[0 0.5]);
outliersleft = excludedata(1:entries,left,'range',[0 0.5]);
outliersright = excludedata(1:entries,right,'range',[0 0.5]);
outliersrecall = excludedata(1:entries,recall,'range',[0 0.5]);

endtime = max(time(1:entries));

%for i=1:entries
subplot(4,1,1)
    plot(time(1:entries),thedata(1:entries,1),'r');
    title('Acceleration (m/s^2) as reported by XSens');
    xlabel('time (s)')
    ylabel('acceleration (m/s^2)')
    hold on
    plot(time(1:entries),thedata(1:entries,2),'g');
    hold on
    plot(time(1:entries),thedata(1:entries,3),'b');
    axis([0,endtime,-20,20])
    legend('x-acc.','y-acc.','z-acc.','Location','Southwest');
    
subplot(4,1,2)
    plot(time(1:entries),thedata(1:entries,4),'r');
    title('Pitch (degrees) and Roll (degrees) as reported by XSens');
    xlabel('time (s)')
    ylabel('degrees')
    hold on
    plot(time(1:entries),thedata(1:entries,5),'g');
    axis([0,endtime,-91,91])
    legend('roll','pitch','Location','Southwest');
    
subplot(4,1,3)
    plot(time(1:entries),thedata(1:entries,6),'r');
    title('Yaw (degrees) as reported by XSens and Heading (degrees) and GPS Heading Accuracy (degrees) as reported by GPS');
    xlabel('time (s)')
    ylabel('degrees')
    hold on
    plot(time(1:entries),thedata(1:entries,20),'g');
    %legend('Yaw from XSens','Heading from GPS');
    hold on
    plot(time(1:entries),thedata(1:entries,22),'b');
    axis([0,endtime,0,361])
    legend('Yaw','Head.','Head. Acc.','Location','Southwest');
    
    subplot(4,1,4)
    %this weird scheme with outliers is to prevent showing the lines when
    %the command is zero (no command)
    plot(time(outliersforward),forward(outliersforward),'^g');
    hold on
    plot(time(outliersstop),stop(outliersstop),'sr');
    hold on
    plot(time(outliersleft),left(outliersleft),'<');
    hold on
    plot(time(outliersright),right(outliersright),'>');
    hold on
    plot(time(outliersrecall),recall(outliersrecall),'*k');
    title('Command (binary) as reported by Tone and Vibration Generator');
    xlabel('time (s)')
    ylabel('Command');
    axis([0,endtime,0,1.4])
 
    legend('Forward','Stop','Left','Right','Recall','Location','Southwest');
%end

%plot(East(1:entries),North(1:entries)); %plot the path of GPS
%xlabel('E-W position(m)');
%ylabel('N-S position(m)');
%title(filename);



hold on	



% -------------

% -------------------------------

% Set for 1/22/09 -- map1

% #S
% 32.5944329 -85.4975336
% #D
% #32.5942707 -85.4973327
% #B
% #32.5944182 -85.4973237
% #C
% 32.5944153 -85.4971414
% #A
% #32.5945101 -85.4972142


% 
%  hold on
% [ENU(:,i+1)] = wgslla2enu(32.5944329, -85.4975336,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+1) , ENU(2,i+1) , 'o','MarkerSize',10);
% text( ENU(1,i+1) + 2, ENU(2,i+1),'S');
% 
%   hold on
%   [ENU(:,i+2)] = wgslla2enu(32.5945101, -85.4972142,0, Latgpsm(1),Longgpsm(1),0);
%   plot( ENU(1,i+2) , ENU(2,i+2) , 'o','MarkerSize',10);
%   text( ENU(1,i+2) + 2, ENU(2,i+2),'A');
% 
%   hold on
%  [ENU(:,i+3)] = wgslla2enu(32.5944182, -85.4973237,0, Latgpsm(1),Longgpsm(1),0);
%  plot( ENU(1,i+3) , ENU(2,i+3) , 'o','MarkerSize',10);
%  text( ENU(1,i+3) + 2, ENU(2,i+3),'B'); 
% 
%   hold on
%  [ENU(:,i+4)] = wgslla2enu(32.5944153, -85.4971414,0, Latgpsm(1),Longgpsm(1),0);
%  plot( ENU(1,i+4) , ENU(2,i+4) , 'o','MarkerSize',10);
%  text( ENU(1,i+4) + 2, ENU(2,i+4),'C');
% 
%  hold on
%  [ENU(:,i+5)] = wgslla2enu(32.5942707, -85.4973327,0, Latgpsm(1),Longgpsm(1),0);
%  plot( ENU(1,i+5) , ENU(2,i+5) , 'o','MarkerSize',10);
%  text( ENU(1,i+5) + 2, ENU(2,i+5),'D');

% 	

% %
% hold on
%  [ENU(:,i+3)] = wgslla2enu(33.7268955, -85.7868446,0, Latgpsm(1),Longgpsm(1),0);
%  plot( ENU(1,i+3) , ENU(2,i+3) , 'o','MarkerSize',10);
%  text( ENU(1,i+3) + 2, ENU(2,i+3),'S'); 
% 
%  hold on
% [ENU(:,i+1)] = wgslla2enu(33.727394, -85.7874239,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+1) , ENU(2,i+1) , 'o','MarkerSize',10);
% text( ENU(1,i+1) + 2, ENU(2,i+1),'A');
% 
%   hold on
%   [ENU(:,i+2)] = wgslla2enu(33.7275375, -85.7876402,0, Latgpsm(1),Longgpsm(1),0);
%   plot( ENU(1,i+2) , ENU(2,i+2) , 'o','MarkerSize',10);
%   text( ENU(1,i+2) + 2, ENU(2,i+2),'B');
%   
%   % front of the RV
%     hold on
%   [ENU(:,i+5)] = wgslla2enu(33.7273205, -85.7874409,0, Latgpsm(1),Longgpsm(1),0);
%   plot( ENU(1,i+5) , ENU(2,i+5) , 'o','MarkerSize',10);
%  text( ENU(1,i+5) + 2, ENU(2,i+5),'D');
%   

%[ENU(:,i+1)] = wgslla2enu(33.7192124, -85.776869,0, Latgpsm(1),Longgpsm(1),0);
%plot( ENU(1,i+1) , ENU(2,i+1) , 'o','MarkerSize',10);
%text( ENU(1,i+1) + 2, ENU(2,i+1),'A');

% [ENU(:,i+1)] = wgslla2enu(33.719261, -85.776989,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+1) , ENU(2,i+1) , 'o','MarkerSize',10);
% text( ENU(1,i+1) + 2, ENU(2,i+1),'A');
% 
% hold on	
% 
% 
% [ENU(:,i+2)] = wgslla2enu(33.7190803, -85.7767454,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+2) , ENU(2,i+2) , 'o','MarkerSize',10);
% text( ENU(1,i+2) + 2, ENU(2,i+2),'B');
% 
% 
% hold on
% [ENU(:,i+3)] = wgslla2enu(33.7190347, -85.7773287,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+3) , ENU(2,i+3) , 'o','MarkerSize',10);
% text( ENU(1,i+3) + 2, ENU(2,i+3),'S');
% 
% hold on
% [ENU(:,i+4)] = wgslla2enu(33.7194998, -85.7771644 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+4) , ENU(2,i+4) , 'o','MarkerSize',10);
% text( ENU(1,i+4) + 2, ENU(2,i+4),'C');
% 
% hold on
% [ENU(:,i+5)] = wgslla2enu(33.7192729, -85.7776098,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+5) , ENU(2,i+5) , 'o','MarkerSize',10);
% text( ENU(1,i+5) + 2, ENU(2,i+5),'D');
% 
% %
% hold on
% [ENU(:,i+6)] = wgslla2enu(33.7188063, -85.7771076,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+6) , ENU(2,i+6) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+6) + 2, ENU(2,i+6),'S2');

% hold on
% [ENU(:,i+2)] = wgslla2enu(33.7391497, -85.7665212,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+2) , ENU(2,i+2) , 'o','MarkerSize',10);
% text( ENU(1,i+2) + 2, ENU(2,i+2),'B2');
% 
% hold on
% [ENU(:,i+1)] = wgslla2enu(33.7391487, -85.7663136,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+1) , ENU(2,i+1) , 'o','MarkerSize',10);
% text( ENU(1,i+1) + 2, ENU(2,i+1),'B2_2');
% 
% hold on	
% 
% [ENU(:,i+2)] = wgslla2enu(33.7393256, -85.7682352,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+2) , ENU(2,i+2) , 'o','MarkerSize',10);
% text( ENU(1,i+2) + 2, ENU(2,i+2),'B_building2');
% 
% hold on
% [ENU(:,i+3)] = wgslla2enu(33.7392631, -85.7674762,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+3) , ENU(2,i+3) , 'o','MarkerSize',10);
% text( ENU(1,i+3) + 2, ENU(2,i+3),'B_building3');
% 
% hold on
% [ENU(:,i+4)] = wgslla2enu(33.7400606, -85.768554 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+4) , ENU(2,i+4) , 'o','MarkerSize',10);
% text( ENU(1,i+4) + 2, ENU(2,i+4),'B_building');
% 
% 
% hold on
% [ENU(:,i+5)] = wgslla2enu(33.7396228, -85.768636,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+5) , ENU(2,i+5) , 'o','MarkerSize',10);
% text( ENU(1,i+5) + 2, ENU(2,i+5),'B_car');
% 
% %
% hold on
% [ENU(:,i+6)] = wgslla2enu(33.7391577, -85.7663518 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+6) , ENU(2,i+6) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+6) + 2, ENU(2,i+6),'C2');
% 
% hold on
% [ENU(:,i+7)] = wgslla2enu(33.7390201, -85.7667609 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+7) , ENU(2,i+7) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+7) + 2, ENU(2,i+7),'D2');
% 
% hold on
% [ENU(:,i+8)] = wgslla2enu(33.7386279, -85.7665366 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+8) , ENU(2,i+8) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+8) + 2, ENU(2,i+8),'entryway');
% 
% hold on
% [ENU(:,i+9)] = wgslla2enu(33.7387073, -85.7666589,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+9) , ENU(2,i+9) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+9) + 2, ENU(2,i+9),'lastcorner');
% 
% hold on
% [ENU(:,i+10)] = wgslla2enu(33.7393451, -85.76836 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+10) , ENU(2,i+10) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+10) + 2, ENU(2,i+10),'S1');
% 
% hold on
% [ENU(:,i+11)] = wgslla2enu(33.7394148, -85.7681521 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+11) , ENU(2,i+11) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+11) + 2, ENU(2,i+11),'S2');
% 
% hold on
% [ENU(:,i+12)] = wgslla2enu(33.7391908, -85.7672117  ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+12) , ENU(2,i+12) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+12) + 2, ENU(2,i+12),'S3');
% 
% 
% hold on
% [ENU(:,i+13)] = wgslla2enu(33.7389422, -85.7666286 ,0, Latgpsm(1),Longgpsm(1),0);
% plot( ENU(1,i+13) , ENU(2,i+13) , 'o','MarkerSize',10); %E, map 2
% text( ENU(1,i+13) + 2, ENU(2,i+13),'S4');

%overlay the goal points


