function [] = plot_dog(filename, mapfile)

thedata = load(filename);

A = size(thedata)
attributes = A(2)
entries = A(1)

thepoints = load(mapfile)
B = size(thepoints)
numpoints = B(1)
 
lat_index = 13
lon_index = 12
com_index = 33
hold on
%overlay the dog's path
%plot(thedata(1:entries,12),thedata(1:entries,13))

Latgpsmu = []; %create vectors to store this data
Longgpsmu = [];

%filter out the really crazy stuff

for i=1:entries
    if ( thedata(i,lon_index) > -80 | thedata(i,lon_index) < -90 | thedata(i,lat_index) > 40 | thedata(i,lat_index) < 30)
        'rejected'
        i
    else
        Latgpsmu = [Latgpsmu thedata(i,lat_index)];
        Longgpsmu = [Longgpsmu thedata(i,lon_index)];
    end
                
end

C = size(Latgpsmu);
entries = C(2)

for i=1:entries
        Latgpsm(i)=Latgpsmu(i);
        Longgpsm(i)=Longgpsmu(i);

        Latgpsm(i);
        Longgpsm(i);
        %This program takes the degree values for lat and long and converts
        %them to east, north, up in meters
        [ENU(:,i)]=wgslla2enu(Latgpsm(i),Longgpsm(i),0,Latgpsm(1),Longgpsm(1),0);
        North(i)=ENU(2,i);
        East(i)=ENU(1,i);                    
end




plot(East(1:entries),North(1:entries)); %plot the path of GPS
xlabel('E-W position(m)');
ylabel('N-S position(m)');
%title(filename);

hold on
plot(1000,1000,'^g','MarkerSize',10); %forward tone
hold on
plot(1000,1000,'sr','MarkerSize',10); %stop tone
hold on
plot(1000,1000,'<'); %left vibrate
hold on
plot(1000,1000,'>'); %right vibrate
hold on
plot(1000,1000,'*k'); %recall tone
legend('GPS Position', 'forward','stop','left vibrate','right vibrate','recall','Location','EastOutside');



hold on	

min(East)
max(East)
min(North)
max(North)
axis([min(East) - 5,max(East) + 5, min(North)-5, max(North)+5]);

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

labels = ['S','A','B','C','D','E','F','G']

for k = 1:numpoints
     hold on
[ENU(:,i+k)] = wgslla2enu(thepoints(k,1), thepoints(k,2),0, Latgpsm(1),Longgpsm(1),0);
plot( ENU(1,i+k) , ENU(2,i+k) , 'o','MarkerSize',10);
text( ENU(1,i+k) + 2, ENU(2,i+k), labels(k));
end

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



for i = 1:entries
    if (thedata(i,com_index) == 49) %forward
        hold on
        plot(East(i),North(i),'^g','MarkerSize',10)
    elseif (thedata(i,com_index) == 51) %stop
        hold on
        plot(East(i),North(i),'sr','MarkerSize',10)
    elseif (thedata(i,com_index) == 52) % left vibrator
        hold on
        plot(East(i),North(i),'<','MarkerSize',14)
    elseif (thedata(i,com_index) == 55) %right vibrator
        hold on
        plot(East(i),North(i),'>','MarkerSize',14)   
    elseif (thedata(i,com_index) == 57) %recall
        hold on
        plot(East(i),North(i),'*k')        
    end
        
end

