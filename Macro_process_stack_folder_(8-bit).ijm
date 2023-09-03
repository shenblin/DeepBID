#@ File (label = "Input directory", style = "directory") input

list = getFileList(input);
list = Array.sort(list);
for (i = 0; i < list.length; i++) {
	open(input + File.separator + list[i]);
	run("8-bit");
	run("Save");
	run("Close All");
}

