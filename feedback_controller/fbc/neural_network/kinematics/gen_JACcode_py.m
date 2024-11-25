function [J, codestr] = gen_JACcode_py(v, q, filename, comment)
% Erhan Oztop June 2021
% Create a Jacobian matrix code (python) for delv/delq  (v is a length column vector) 
functionname = 'compute_jac';
vpa_digits = 8;
n = length(q);
m = length(v);
if exist('filename','var')   %  filename can be ../xx/yy/zzz/abc.py
    dot = strfind(filename,'.');
    slash = strfind(filename,'/');  % be careful in windows folders
    dot = dot(end);     % last dot   xxxx.py
    if isempty(slash), slash = 1; else, slash = slash(end); end
    classname = filename(slash+1:dot-1);
else
    classname = 'POSJAC_CLASS';
end



syms x;
for i = 1:m
    for j = 1:n
        J(i,j) = x;
    end
end



for i = 1:m
    for j = 1:n
        J(i,j) = diff(v(i), q(j));
    end
end

args='';
for j=1:n
    if (j<n)
        args = sprintf('%s%s, ',args,char(q(j)));
    else
        args = sprintf('%s%s',args,char(q(j)));
    end
end

% codestr=sprintf('%% POSITION JACOBIAN\n');
% codestr = sprintf('%sfunction J = %s(%s)\n',codestr,functionname,args);
% sprintf('%s    J=zeros(%d,%d);\n', codestr, 3, n);
% for i = 1:3
%     for j = 1:n
%         codestr=sprintf('%s    J(%d,%d) = %s;\n',codestr,i,j,char(vpa(J(i,j))));
%     end
% end
% codestr = sprintf('%s%%end of computePosJac\n', codestr);
% 
% 
% if exist('filename','var')
%     fid= fopen(filename,'w');
%     if (fid>0)
%         fprintf(fid, '%s\n',codestr);
%         fclose(fid);
%         fprintf('Position Jacobian code is written to %s \n', filename);
%     else
%        fprintf('Position Jacobian code file %s cannot be created \n', filename); 
%     end
% end
% %===


args='';
for j=1:n
    if (j<n)
        args = sprintf('%s%s, ',args,char(q(j)));
    else
        args = sprintf('%s%s',args,char(q(j)));
    end
end

if ~exist('comment','var'), comment = ' Analytic Jacobian'; end
codestr=sprintf('# %s\n',comment);
codestr = sprintf('%s# Try: [J,~]=%s.%s(0,0, 0,0,0,0,0,0,0, 0)\n',codestr,classname,functionname);
codestr = sprintf('%simport numpy as np\n\n', codestr);
codestr = sprintf('%sclass %s:\n', codestr,classname);
codestr = sprintf('%s\t@staticmethod\n\tdef %s(%s):\n',codestr,functionname,args);
codestr = sprintf('%s\t\tJ = np.zeros([%d,%d]);\n', codestr, m, n);
for i = 1:m
    for j = 1:n
        codestr=sprintf('%s\t\tJ[%d,%d] = %s\n',codestr,i-1,j-1,addnp(char(vpa(J(i,j),vpa_digits))));
    end
end
codestr = sprintf('%s\t\treturn J\n',codestr);
codestr = sprintf('%s\t#end of compute Jacobian\n', codestr);
codestr = sprintf('%s#end of class %s\n', codestr,classname);

codestr = sprintf('%sdef main():\n\tprint %s.%s(0,0, 0,0,0,0, 0,0,0, 0)\n\n',codestr,classname,functionname);
codestr = sprintf('%sif __name__ == ''__main__'':\n\tmain()\n',codestr);


if exist('filename','var')
    fid= fopen(filename,'w');
    if (fid>0)
        fprintf(fid, '%s\n',codestr);
        fclose(fid);
        fprintf('Position Jacobian code is written to %s \n', filename);
    else
       fprintf('Position Jacobian code %s cannot be created \n', filename); 
    end
end

function s = addnp(s)
    s = replace(s,'cos','np.cos');
    s = replace(s,'sin','np.sin');
    s = replace(s,'pi','np.pi');