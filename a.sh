# List all Docker images and save their names to a file
#sudo docker images --format "{{.Repository}}:{{.Tag}}" > names.txt

# Display the contents of the file
#cat names.txt

#sudo docker save -o postgres.tar registry.gitlab.com/everything-group/x23/postgres:latest || { echo "Permission denied. Please check your Docker permissions or run with appropriate privileges."; exit 1; }

chmod 777 ./
# sudo docker save -o postgres.tar registry.gitlab.com/everything-group/x23/postgres:latest


sudo docker save -o per.tar registry.gitlab.com/everything-group/x23/jcfx:latest

scp ixotc@192.229.23.244:/home/ixotc/algo/rouhollah_account/*.tar .

docker load -i per.tar