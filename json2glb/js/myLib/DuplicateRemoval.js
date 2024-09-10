function DuplicateRemoval(){
    this.resourceManager;
    this.meshs;
    this.index;
}
DuplicateRemoval.prototype={
    process:function (resourceManager) {
        this.resourceManager=resourceManager;
        this.meshs=this.resourceManager.meshs;
        //this.computeIndex();
        this.test();
    },
    computeIndex:function(){
        this.index=[];
        for(var i=0;i<this.meshs.length;i++){
            test.push(i);
        }
        for(i=0;i<meshs.length;i++){
            for(var j=0;j<i;j++){

                if(meshs[i].material===meshs[j].material){

                }
            }
        }
    },
    test:function () {
        var meshs=this.resourceManager.meshs;
        var test=[];
        var n=0;
        for(i=0;i<meshs.length;i++){
            test.push(i);
        }
        for(i=0;i<meshs.length;i++){
            for(j=0;j<i;j++){
                test.push(i);
                if(this.similarGeometry(meshs[i].geometry,meshs[j].geometry)){
                    test[j]=i; n++;
                }
            }
        }
        console.log("geometry:",test,n,n/test.length);

        var test2=[];
        var n2=0;
        for(var i=0;i<meshs.length;i++){
            test2.push(i);
        }
        for(i=0;i<meshs.length;i++){
            for(var j=0;j<i;j++){
                test2.push(i);
                if(meshs[i].material===meshs[j].material){
                    test2[j]=i; n2++;
                }
            }
        }
        console.log("material:",test2,n2,n2/test2.length);
    },
    similarGeometry(geometry1,geometry2){
        if(!similarBuffer(geometry1.index,geometry2.index))return false;
        if(!similarBuffer(geometry1.attributes.normal,geometry2.attributes.normal))return false;
        if(!similarBuffer(geometry1.attributes.position,geometry2.attributes.position))return false;
        if(!similarBuffer(geometry1.attributes.uv,geometry2.attributes.uv))return false;
        return false;
        function similarBuffer(buffer1,buffer2) {
            if(buffer1===null&&buffer2!==null)return false;
            else if(buffer2===null&&buffer1!==null)return false;
            else if(buffer1&&buffer2){
                if(!buffer1.array||!buffer2.array){
                    alert("缓冲区无array对象")
                    return false;
                }
                if(buffer1.array.length!==buffer2.array.length)
                    return false;
                for(var ii=0;ii<buffer1.array.length;ii++){
                    if(buffer1.array[ii]!==buffer2.array[ii])
                        return false;
                }
            }
            return true;
        }
    },
}
