#!/bin/bash

quickcd_config=( 
	'xglue /home/zyh/python/github.com/microsoft/CodeXGLUE/Text-Code/NL-code-search-WebQuery'
)

goto_cd () {
	for ((i = 0; i < ${#quickcd_config[@]}; i++)); do 
		local shortcut=${quickcd_config[$i]%% *} 
		local path=${quickcd_config[$i]#* }
      		if [ "$1" == "$shortcut" ]; then
			\cd "$path"
		fi	       
	done
}

goto_help() { 
	for ((i = 0; i < ${#quickcd_config[@]}; i++)); do 
		local shortcut=${quickcd_config[$i]%% *} 
		local path=${quickcd_config[$i]#* } 
		echo -e "$shortcut\t=>\t$path" 
	done
	echo -e "example: tap 'goto ${quickcd_config[0]%% *}' to run 'cd ${quickcd_config[0]#* }'" 
}

if [ $# -ne 1 ]; then
	goto_help
	return 1
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ "$1" == "help" ]; then
	goto_help
	return 0
fi

goto_cd "$1"

return 0
