function showframe(F)

x=F(:,1);
y=F(:,2);
z=F(:,3);

W=diag([1 1 1]);
wx=W(:,1);
wy=W(:,2);
wz=W(:,3);

plot3([0,wx(1)],[0,wx(2)], [0,wx(3)],'r--','Linewidth',5); hold on;
plot3([0,wy(1)],[0,wy(2)], [0,wy(3)],'g--','Linewidth',5);  
plot3([0,wz(1)],[0,wz(2)], [0,wz(3)],'b--','Linewidth',5);  

plot3([0,x(1)],[0,x(2)], [0,x(3)],'r-','Linewidth',2); hold on;
plot3([0,y(1)],[0,y(2)], [0,y(3)],'g-','Linewidth',2);  
plot3([0,z(1)],[0,z(2)], [0,z(3)],'b-','Linewidth',2);  
axis([-1 1 -1 1 -1 1])
xlabel('Wx'); ylabel('Wy'); zlabel('Wz');