#include<stdio.h>

#include<pcap.h>

#include<string.h>

#include <arpa/inet.h>

#include <sys/types.h>

#include <sys/_endian.h>

#include<pthread.h>

#include <stdlib.h>

#include <netdb.h>

#include <sys/socket.h>

#include <netinet/in.h>

#include <unistd.h>

#include <stdlib.h>

#include "Header.h"



//#define PACP_ERRBUF_SIZE 256



void callback(u_char * userarg, const struct pcap_pkthdr * pkthdr, const u_char * packet) {

    pcap_dump(userarg, pkthdr, packet);

    printf("--------------------------------------\n");

    struct MacHeader *macHeader = (struct MacHeader*)packet;

    printf("%d %d\n",pkthdr->caplen,pkthdr->len);

    pthread_t t = pthread_self();

    printf("child thread tid = %ld\n", t->__sig);

    printf("����������·��\n");

    printf("Ŀ�� Mac��ַ: %02x:%02x:%02x:%02x:%02x:%02x\n",macHeader->des[0]&0x0ff,macHeader->des[1]&0x0ff,macHeader->des[2]&0x0ff,macHeader->des[3]&0x0ff,macHeader->des[4]&0x0ff,macHeader->des[5]&0x0ff);

    printf("Դ Mac��ַ: %02x:%02x:%02x:%02x:%02x:%02x\n",macHeader->source[0]&0x0ff,macHeader->source[1]&0x0ff,macHeader->source[2]&0x0ff,macHeader->source[3]&0x0ff,macHeader->source[4]&0x0ff,macHeader->source[5]&0x0ff);



    printf("%04x",ntohs(macHeader->type));

    printf("\n");

    printf("���������\n");

    struct IpPackage ipPackage = *((struct IpPackage*)(packet+14));

    struct in_addr sourceIp = {ipPackage.source};

    struct in_addr desIp = {ipPackage.des};

    int *proctoal = (int *)&ipPackage.protocal;

    int *verandlen = (int *)&ipPackage.verandlen;

    printf("Э���ֶ�ֵΪ%d\n",*proctoal&0xff);

    printf("�ײ�����Ϊ%dB\n",((*verandlen)&0xf)*4);

    printf("���ݲ��ֳ���Ϊ%dB\n",ntohs(ipPackage.packageLen) - ((*verandlen)&0xf)*4);

    printf("Ƭƫ��Ϊ:%d\n",ntohs(ipPackage.offset&0x1fff));

//    printf("ԴipΪ�� ip=%s\n",inet_ntoa(sourceIp));

//    printf("Ŀ��ipΪ�� ip=%s\n",inet_ntoa(desIp));

    printf("\n");

    printf("���������\n");

    

    if ((*proctoal&0xff) == 6) {

        printf("tcp��\n");

        struct TcpPackage *tcpPackage = (struct TcpPackage*)(packet+14+((*verandlen)&0xf)*4);

        printf("Դ�˿�Ϊ%d    ",ntohs(tcpPackage->source));

        printf("Ŀ�Ķ˿�Ϊ%d   ",ntohs(tcpPackage->des));

        printf("ʱ��Ϊ%ld    ",pkthdr->ts.tv_sec);

        printf("ԴipΪ��ip=%s         ",inet_ntoa(sourceIp));

        printf("Ŀ��ipΪ��ip=%s        ",inet_ntoa(desIp));

        unsigned syn = ntohl(tcpPackage->number);

        printf("���Ϊ%u     ",syn);

        printf("ȷ�Ϻ�Ϊ%u     ",ntohl(tcpPackage->confirmNumber));

        printf("���ڴ�СΪ%d    ",ntohs(tcpPackage->windowSize));

        printf("FIN=%d     ",(tcpPackage->others&0x0100) != 0);

        printf("ACK=%d     ",(tcpPackage->others&0x1000) != 0);

        printf("SYN=%d     ",(tcpPackage->others&0x0200) != 0);

        printf("���ݲ��ֳ���%d         ",ntohs(ipPackage.packageLen) - ((*verandlen)&0xf)*4-((tcpPackage->others&0x00f0)>>4)*4);

        printf("�ײ�����%d\n",((tcpPackage->others&0x00f0)>>4)*4);

    } else if((*proctoal&0xff) == 17) {

        printf("udp��");

        struct UdpPackage *udpPackage = (struct UdpPackage*)(packet+14+((*verandlen)&0xf)*4);

        printf("ԴipΪ��ip=%s         ",inet_ntoa(sourceIp));

        printf("Ŀ��ipΪ��ip=%s        ",inet_ntoa(desIp));

        printf("Դ�˿�Ϊ%d    ",htons(udpPackage->source));

        printf("Ŀ�Ķ˿�Ϊ%d   ",htons(udpPackage->des));

        printf("���ݲ��ֳ���%d\n   ",htons(udpPackage->udpLength)-8);

        

    } else {

        printf("������ ip��");

    }

 

    printf("++++++++++++++++++++++++++++++++++++++++\n");

}



struct in_addr getIp() {

    static struct in_addr addr = {0};

    if (addr.s_addr == 0) {

        char hname[128];

        struct hostent *hent;

        gethostname(hname, sizeof(hname));

        hent = gethostbyname(hname);

        addr = *(struct in_addr*)(hent->h_addr_list[0]);

    }

    return addr;

}



int getSerive(int port) {

    int i;

    switch (port) {

        case 21:

            i = 0;

            break;

        case 23:

            i = 1;

            break;

        case 25:

            i = 2;

            break;

        case 80:

            i = 3;

            break;

        case 443:

            i = 4;

            break;

        case 53:

            i = 0;

            break;

        case 69:

            i = 1;

            break;

        case 161:

            i = 2;

            break;

        case 67:

            i=3;

            break;

        default:

            i = -1;

            break;

    }

    return i;

}





void saveDataPackage(struct DataPackageQueue*queue,struct DataPackae* dataPackage) {

    struct in_addr addr = getIp();

    ushort port;

    int flag;

    struct DataPackae* temp = NULL;

    struct DataPackae* head = NULL;

    flag = dataPackage->sourceIp.s_addr == addr.s_addr ? 1:0;

    port = flag == 1 ? dataPackage->desPort:dataPackage->sourcePort;

    unsigned int ip = flag == 1 ? dataPackage->desIp.s_addr:dataPackage->sourceIp.s_addr;

 //   printf("Դ ip = %s  Դ�˿�Ϊ %d  flag = %d port = %d  %d\n",inet_ntoa(dataPackage->sourceIp),dataPackage->sourcePort,flag,port,dataPackage->dataLength);

 //   printf("Ŀ�� ip = %s Ŀ�Ķ˿�Ϊ%d\n",inet_ntoa(dataPackage->desIp),dataPackage->desPort);

    int i = getSerive(port);

    if (i == -1) {

        printf("δ֪Ӧ�ò����  port = %d\n",port);

        free(dataPackage);

        return;

    }

    int j = (dataPackage->sourceIp.s_addr ^ dataPackage->desIp.s_addr) % 256;

    if (flag == 1) {

        queue[i].uploadCount += dataPackage->dataLength;

        queue[i].uploadDuringCount += dataPackage->dataLength;

        queue[i].package[j].uploadCount += dataPackage->dataLength;

        head = (queue[i].package[j].upload);

        if (queue[i].package[j].upload == NULL) {

            queue[i].package[j].upload = dataPackage;

            return;

        }

    } else if(flag == 0) {

        queue[i].downCount += dataPackage->dataLength;

        queue[i].downDuringCount += dataPackage->dataLength;

        queue[i].package[j].downCount += dataPackage->dataLength;

        head = (queue[i].package[j].down);

        if (queue[i].package[j].down == NULL) {

            queue[i].package[j].down = dataPackage;

            return;

        }

    }



    temp = head;

    unsigned int findIp = flag == 1 ? dataPackage->desIp.s_addr:dataPackage->sourceIp.s_addr;

    while (temp) {

        if (findIp == ip) {

            dataPackage->next = temp->next;

            temp->next = dataPackage;

            break;

        }

        if (temp->next == NULL) {

            temp->next = dataPackage;

            break;

        }

        temp = temp->next;

    }

    

}





void freeDataPackae(struct DataPackae* dataPackage) {

    struct DataPackae* temp;

    printf("��ʼ�ͷ�\n");

    while(dataPackage) {

        temp = dataPackage;

        printf("ԴipΪ%s       Ŀ��ipΪ%s     ", inet_ntoa(dataPackage->sourceIp),inet_ntoa(dataPackage->desIp));

        printf("Ŀ�Ķ˿�Ϊ%d    Դ�˿�%d     ",dataPackage->sourcePort,dataPackage->desPort);

        printf("ipͷ������%d    �����ͷ������%d     ���ݲ��ֳ���%d\n",dataPackage->ipHeaderLength,dataPackage->headerLength,dataPackage->dataLength);

        dataPackage = dataPackage->next;

        free(temp);

    }

    printf("�����ͷ�\n");

}



void freeQueue() {

     for (int i = 0; i < 5; i++) {

        printf("Ӧ�ò�Э��:%s   �ܵ��ϴ���%ld   �ܵ�������%ld   һ��ʱ��Ƭ���ϴ���%ld    һ��ʱ��Ƭ��������%ld\n",tcpName[i]

               ,tcpQueue[i].uploadCount,tcpQueue[i].downCount,tcpQueue[i].uploadDuringCount,

               tcpQueue[i].downDuringCount);

        tcpQueue[i].downDuringCount = 0;

        tcpQueue[i].uploadDuringCount = 0;

        if (tcpQueue[i].downDuringCount == 0 && tcpQueue[i].uploadDuringCount == 0) {

            continue;

        }



        for (int j = 0; j < 256; j++) {

            struct Package tcp = tcpQueue[i].package[j];

            if (tcp.downCount != 0) {

                freeDataPackae(tcp.down);

                tcp.downCount = 0;

            }

            tcp.down = NULL;

            if (tcp.uploadCount !=0) {

                freeDataPackae(tcp.upload);

                tcp.uploadCount = 0;

            }

            tcp.upload = NULL;

            

        }

    }

    

    for (int i = 0; i < 5; i++) {

        printf("Ӧ�ò�Э��:%s   �ܵ��ϴ���%ld   �ܵ�������%ld   һ��ʱ��Ƭ���ϴ���%ld    һ��ʱ��Ƭ��������%ld\n",utpName[i]

               ,udpQueue[i].uploadCount,udpQueue[i].downCount,udpQueue[i].uploadDuringCount,

               udpQueue[i].downDuringCount);

        udpQueue[i].downDuringCount = 0;

        udpQueue[i].uploadDuringCount = 0;

        if (udpQueue[i].downDuringCount == 0 && udpQueue[i].uploadDuringCount == 0) {

            continue;

        }

        for (int j = 0; j < 256; j++) {

            struct Package udp = udpQueue[i].package[j];

            if (udp.downCount != 0) {

                freeDataPackae(udp.down);

                udp.downCount = 0;

            }

            udp.down = NULL;

            if (udp.uploadCount != 0) {

                freeDataPackae(udp.upload);

                udp.uploadCount = 0;

                

            }

            udp.upload = NULL;

        }

    }

}



void analysisCallback(u_char * userarg, const struct pcap_pkthdr * pkthdr, const u_char * packet) {

    static long seconds = -1;

    static int during = 60;

    if (seconds == -1) {

        seconds = pkthdr->ts.tv_sec;

        during = atoi((char*)userarg);

    }

    if((pkthdr->ts.tv_sec - seconds) >= during) {

        printf("%ld �� %ld\n",seconds,pkthdr->ts.tv_sec);

        seconds = pkthdr->ts.tv_sec;

        freeQueue();

    }

    

    //struct MacHeader *macHeader = (struct MacHeader*)packet;

    struct IpPackage ipPackage = *((struct IpPackage*)(packet+14));

    int *proctoal = (int *)&ipPackage.protocal;

    struct DataPackae* dataPackage = NULL;

    dataPackage = (struct DataPackae*)malloc(sizeof(struct DataPackae));

    memset(dataPackage, 0, sizeof(struct DataPackae));

    if(dataPackage == NULL) {

        printf("error");

        exit(0);

    }

    struct in_addr sourceIp = {ipPackage.source};

    struct in_addr desIp = {ipPackage.des};



    dataPackage->desIp.s_addr = desIp.s_addr;

    dataPackage->sourceIp.s_addr = sourceIp.s_addr;

    

    int *verandlen = (int *)&ipPackage.verandlen;

    dataPackage->ipHeaderLength = ((*verandlen)&0xf)*4;

    dataPackage->ts = pkthdr->ts;

    dataPackage->next = NULL;

    if ((*proctoal&0xff) == 6) {

        struct TcpPackage *tcpPackage = (struct TcpPackage*)(packet+14+((*verandlen)&0xf)*4);

        dataPackage->desPort = htons(tcpPackage->des);

        dataPackage->sourcePort = htons(tcpPackage->source);

        dataPackage->headerLength = ((tcpPackage->others&0x00f0)>>4)*4;

        dataPackage->dataLength = htons(ipPackage.packageLen) - ((*verandlen)&0xf)*4-((tcpPackage->others&0x00f0)>>4)*4;

        if (dataPackage->dataLength == 0) {

            free(dataPackage);

        }else {

           saveDataPackage(tcpQueue,dataPackage);

        }

    } else if ((*proctoal&0xff) == 17) {

        struct UdpPackage *udpPackage = (struct UdpPackage*)(packet+14+((*verandlen)&0xf)*4);

        dataPackage->desPort = htons(udpPackage->des);

        dataPackage->sourcePort = htons(udpPackage->source);

        dataPackage->headerLength = 8;

        dataPackage->dataLength = htons(udpPackage->udpLength)-8;

        if (dataPackage->dataLength == 0) {

            free(dataPackage);

        }else {

            saveDataPackage(udpQueue,dataPackage);

        }

    } else {

        printf("����Э���ֶ�%d\n",*proctoal&0xff);

        free(dataPackage);

    }

}









int main(int argc,char*argv[])

{

    

 

    

    

    

    

    int flag = atoi(argv[1]);

    if (flag == 1) {

        char *device;//��������򿪵��豸

        char errBuf[PCAP_ERRBUF_SIZE];//���������Ϣ

        device = pcap_lookupdev(errBuf);

        if (device == NULL ) {

            perror("no have net device");

            return 0;

        }

        printf("%s",device);

        bpf_u_int32 ip;

        bpf_u_int32 ma;

        pcap_lookupnet(device, &ip, &ma, errBuf);

        struct in_addr sin_addrIP;

        sin_addrIP.s_addr = ip;

        struct in_addr sin_addrMa = {ma};

        printf("����������ţ� ip=%s\n",inet_ntoa(sin_addrIP));

        printf("������������Ϊ�� ma=%s\n",inet_ntoa(sin_addrMa));

        

        pcap_t *p = pcap_open_live(device, 65535, 0, 0, errBuf);

        int dataType = pcap_datalink(p);

        if (dataType != DLT_EN10MB) {

            printf("������ֻ֧��Ethernet��������·��׼��������׼�������޸Ĵ���·��õ�ip���ݱ��Ĵ���\n");

            return 0;

        }

        if (p == NULL) {

            printf("error11\n");

            return 0;

        }

        printf("��ʼ����\n");

        pcap_dumper_t *k = pcap_dump_open(p, "./test.pcap");

        if (k == NULL) {

            printf("%s",pcap_geterr(p));

        }

        struct bpf_program filter;

        pcap_compile(p, &filter, argv[2], 1, ma);

        pcap_setfilter(p, &filter);

        pcap_loop(p, -1, callback, (u_char*)k);

        pcap_close(p);

    } else if(flag == 2){

        char errBuf[PCAP_ERRBUF_SIZE];//���������Ϣ

        pcap_t*p = pcap_open_offline(argv[2], errBuf);

        if (p == NULL) {

            printf("error\n");

            return 0;

        }

        int dataType = pcap_datalink(p);

        if (dataType != DLT_EN10MB) {

            printf("������ֻ֧��Ethernet��������·��׼��������׼�������޸Ĵ���·��õ�ip���ݱ��Ĵ���\n");

            return 0;

        }

        pcap_loop(p, -1, analysisCallback, (u_char*)argv[3]);

        pcap_close(p);

    }

    

    

    



}
