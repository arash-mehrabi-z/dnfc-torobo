function [codestr] = gen_FKcode_py(M, q, filename)
functionname = 'forward_ik';
n = length(q);
vpa_digits = 8;
if exist('filename','var')   %  filename can be ../xx/yy/zzz/abc.py
    dot = strfind(filename,'.');
    slash = strfind(filename,'/');  % be careful in windows folders
    dot = dot(end);     % last dot   xxxx.py
    if isempty(slash), slash = 1; else, slash = slash(end); end
    classname = filename(slash+1:dot-1);
else
    classname = 'FKIN_CLASS';
end

args='';
for j=1:n
    if (j<n)
        args = sprintf('%s%s, ',args,char(q(j)));
    else
        args = sprintf('%s%s',args,char(q(j)));
    end
end

codestr=sprintf('# FORWARD KINEMATICS \n');
codestr = sprintf('%s# Try: T=%s.%s(0,0, 0,0,0,0,0,0,0, 0) # rotm2quat(T[1:3,1:3])\n',codestr,classname,functionname);
codestr = sprintf('%simport numpy as np\n\n', codestr);
codestr = sprintf('%sclass %s:\n', codestr,classname);
codestr = sprintf('%s\t@staticmethod\n\tdef %s(%s):\n',codestr,functionname,args);
codestr = sprintf('%s\t\tT=np.zeros([%d,%d]);\n', codestr, 4, 4);

for i = 1:4
    for j = 1:4
        codestr=sprintf('%s\t\tT[%d,%d] = %s\n',codestr,i-1,j-1,addnp(char(vpa(M(i,j),vpa_digits))));
    end
end
codestr = sprintf('%s\t\treturn T\n',codestr);
codestr = sprintf('%s\t#end of computeForKin\n', codestr);
codestr = sprintf('%s#end of class %s\n', codestr,classname);

codestr = sprintf('%sdef main():\n\tprint %s.%s(0,0, 0,0,0,0, 0,0,0, 0)\n\n',codestr,classname,functionname);
codestr = sprintf('%sif __name__ == ''__main__'':\n\tmain()\n',codestr);


if exist('filename','var')
    fid= fopen(filename,'w');
    if (fid>0)
        fprintf(fid, '%s\n',codestr);
        fclose(fid);
        fprintf('Forward kinematic code is written to %s \n', filename);
    else
       fprintf('Forward kinematic file %s cannot be created \n', filename); 
    end
end

function s = addnp(s)
    s = replace(s,'cos','np.cos');
    s = replace(s,'sin','np.sin');
    s = replace(s,'pi','np.pi');