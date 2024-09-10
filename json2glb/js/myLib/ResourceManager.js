function ResourceManager() {
    //maps
    // url
    // interest
    // modelUrl
    //models
    // url
    // interest
    // boundingSphere
    //    x,y,z,r
    this.name;
    this.maps;//说明信息
    this.models;//说明信息
    this.meshs;//真正的资源
    this.mapsIndex;
    this.init();
}
ResourceManager.prototype={
    init:function () {
        this.maps=[];
        this.models=[];
    },
    modelsInit:function(IDs,SVs){
        for(i=0;i<this.meshs.length;i++){
            var modelManager=new ModelManager();
            modelManager.fileName=this.name+""+i+".gltf";
            modelManager.interest=IDs[i];
            modelManager.spaceVolume=SVs[i];
            modelManager.boundingSphereSet(this.meshs[i]);
            this.models.push(
                modelManager
            );
        }
    },
    mapsInit:function(mapsIndex,canvass){
        this.mapsIndex=mapsIndex;
        for(i=0;i<mapsIndex.length;i++){
            var modelName=this.name+""+i+".gltf";
            var mapName=this.name+""+i+".jpg";
            var model=this.getModeByName(modelName);
            if(mapsIndex[i]===1){
                var mapManager=new MapManager();
                mapManager.fileName=mapName;
                mapManager.modelName=modelName;
                var mapFileSize=canvass[this.maps.length].toDataURL("image/jpeg").length;
                mapManager.interest=model.spaceVolume/mapFileSize;
                this.maps.push(mapManager);
                model.MapName=mapName;
            }else{
                model.MapName="";
            }
        }
    },
    resourceInfoGet:function () {
        var result={};
        result.maps=this.maps;
        result.models=this.models;
        //result.mapsIndex=this.mapsIndex;//mapsIndex在前端似乎没有用处
        return result;
    },
    getMapByName:function (name) {
        for(i=0;i<this.maps.length;i++){
            if(this.maps[i].fileName===name)
                return this.maps[i];
        }
    },
    getModeByName:function (name) {
        for(i=0;i<this.models.length;i++){
            if(this.models[i].fileName===name)
                return this.models[i];
        }
    },
}
function MapManager() {
    this.fileName;
    this.interest;
    this.modelName;
}
MapManager.prototype={
}
function ModelManager() {
    this.fileName;
    this.interest;
    this.boundingSphere;//x,y,z,r
    this.MapName;//没有纹理贴图时为空
    this.spaceVolume;
}
ModelManager.prototype={
    boundingSphereSet:function (mesh) {
        //mesh.geometry.computeBoundingBox();
        mesh.geometry.computeBoundingSphere();

        mesh.geometry.boundingSphere.applyMatrix4(mesh.matrix);

        this.boundingSphere={
            x:mesh.geometry.boundingSphere.center.x,
            y:mesh.geometry.boundingSphere.center.y,
            z:mesh.geometry.boundingSphere.center.z,
            r:mesh.geometry.boundingSphere.radius
        };
    },
    boundingSphereSet0:function (mesh) {
        mesh.geometry.computeBoundingBox();
        mesh.geometry.computeBoundingSphere();

        var box=mesh.geometry.boundingBox;
        var sx=mesh.scale.x;
        var sy=mesh.scale.y;
        var sz=mesh.scale.z;

        var x0=mesh.geometry.boundingSphere.center.x;
        var y0=mesh.geometry.boundingSphere.center.y;
        var z0=mesh.geometry.boundingSphere.center.z;


        //var matrix=mesh.matrix;
        console.log(mesh.matrix);

        var x=sx*x0 +mesh.position.x;
        var y=sy*y0 +mesh.position.y;
        var z=sz*z0 +mesh.position.z;

        var r=Math.pow(
            Math.pow(sx*(box.max.x-box.min.x),2)+
            Math.pow(sy*(box.max.y-box.min.y),2)+
            Math.pow(sz*(box.max.z-box.min.z),2),
            0.5
        )/2;

        this.boundingSphere={x:x, y:y, z:z, r:r};
    },
}
