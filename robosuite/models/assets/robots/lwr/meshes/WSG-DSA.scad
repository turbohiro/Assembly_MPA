/** WSG-DSA.scad - simplified 3D model of the Weiss Robotics
 *  tactile sensing finger.
 *
 * 11.05.2017 - created
 *
 * (c) 2017 fnh, hendrich@informatik.uni-hamburg.de
 */


 // dimensions are in millimeters, scale by 0.001 for ROS
 eps = 0.1;
 fn = 10;


 translate( [0,0,0] ) DSA_sensing_finger();


 module DSA_sensing_finger()
 {
   // mounting part that connects to the gripper
   color( [0.5,0.5,0.5] )
     translate( [-9/2, 0, 28/2] )
       cube( size=[9, 30, 28], center=true );

   // prism
   color( [0.5,0.5,0.5] )
     translate( [-9,30/2,28] ) rotate( [0,0,-90] )
       prism( 30, 9, 9 );

   // status LED
   color( [0,0,0.8] )
     translate( [0, -30/2, 20] ) rotate( [90,0,0], $fn=fn )
       cylinder( d=5, h=1, center=true );

   // main finger, finger adapter see Schunk manual page 35
   // "Grundbacke" has h=24, w=30, l=18, thickness=9
   //
   hg = 24-9; // height of finger adapter - height of "grundbacke"
   hh = 78.5 - hg;
   color( [0.5,0.5,0.5] )
     translate( [9/2, 0, hg+hh/2] )
       cube( size=[9, 30, hh], center=true );


   // sensors pad, 6x14 cells of 3.4mm size, outer size 51 x 24 x 3
   dx = 12.7 - 9;
   dy = 24;
   dz = 51;
   color( [0.2,0.2,0.2] )
     translate([9+dx/2, 0, 25.3+dz/2] )
       cube( size=[dx, dy, dz], center=true );
}



 module prism(l, w, h){
       polyhedron(
               points=[[0,0,0], [l,0,0], [l,w,0], [0,w,0], [0,w,h], [l,w,h]],
               faces=[[0,1,2,3],[5,4,3,2],[0,4,5,1],[0,3,4],[5,2,1]]
               );

       }
