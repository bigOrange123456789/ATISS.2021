function InDe() {//通过兴趣度调整加载顺序
    this.meshArr;
    this.IDs;
    this.SVs;
}
InDe.prototype={
    process(resourceManager){
        this.meshArr=resourceManager.meshs;
        this.computeID();
        this.sort();//将IDs变为由大到小，对应调整meshaArr中元素顺序
        resourceManager.meshs=this.meshArr;
        resourceManager.modelsInit(this.IDs,this.SVs);
    },
    computeID:function () {//计算兴趣度
        this.IDs=[];
        this.SVs=[];
        for(i=0;i<this.meshArr.length;i++){
            var mesh=this.meshArr[i];
            mesh.geometry.computeBoundingBox();
            var box=mesh.geometry.boundingBox;
            var a=box.max,b=box.min;
            var V=Math.abs(
                (a.x-b.x)*(a.y-b.y)*(a.z-b.z)
            );
            var N=mesh.geometry.attributes.position.count;
            var ID=V/N;
            console.log(mesh.name,V,N,ID);
            this.IDs.push(ID);
            this.SVs.push(V);
        }
    },
    sort:function () {//兴趣度高的优先传输
        //简单选择排序
        var IDs=this.IDs;var temp;
        for(i=0;i<IDs.length-1;i++){
            max=i;
            for(j=i+1;j<IDs.length;j++){
                if(IDs[j]>IDs[max])max=j;
            }
            if(max!==i){
                temp=IDs[i];
                IDs[i]=IDs[max];
                IDs[max]=temp;

                temp=this.meshArr[i];
                this.meshArr[i]=this.meshArr[max];
                this.meshArr[max]=temp;

                temp=this.SVs[i];
                this.SVs[i]=this.SVs[max];
                this.SVs[max]=temp;
            }
        }
    },
}