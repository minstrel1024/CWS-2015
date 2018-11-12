/*
 * CWS.cpp
 *
 *  Created on: 2015-6-30
 *      Author: Sil Alaice
 */

#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<iomanip>
#include<fstream>
#include<vector>
#include<list>
#include<deque>
#include<queue>
#include<set>
#include<map>
#include<iterator>
#include<ctime>

using namespace std;
#define MAXBUFFER 1024
#define MAXTRAIN 100000	//最大训练数据行数
#define MAXK 100
#define MAXSTEPS 7	//训练迭代次数
#define MAXFACTOR 159	//特征向量维数
#define MINWORD 1	//最少多少次算作字典里的词

#define BGRAM 0	//句子起始符编码
#define EGRAM 1	//句子终结符编码
#define INVALIDGRAM -1	//非法字符编码

#define TRAINFILE "train.txt"
#define TESTFILE "test.txt"
#define ANSWERFILE "test.answer.txt"

typedef pair<char, char> pc;
typedef pair< char, pair<char, char> > tc;	//三个char，存放UTF-8类型的汉字(小于3字节的均可存放)
typedef pair<int, int> pii;
typedef pair<bool,pii> pbp;	//(bool a,int b,int c)，表示词b与词c之间是否切分

#define tc(a,b,c) (tc((a),pc((b),(c))))		//将三个char类型转为tc类型
#define pbp(a,b,c) (pbp((a),pii((b),(c))))	//将1个bool和2个int转为pbp类型

int tctoi(tc a)	//将tc类型的UTF-8字符编码为int方便处理
{
	return (unsigned char)(a.first) * 65536 + (unsigned char)(a.second.first) * 256 + (unsigned char)(a.second.second);
}
tc itotc(int a)	//将编码为int的UTF-8还原回tc
{
	tc b;
	b.first = char(a /65536);
	b.second.first = char(a % 65536 / 256);
	b.second.second = char(a % 256);
	return b;
}
int pbitoi(bool a, int b)
{
	if(a)return 2*b;
	else return 2*b+1;
}

vector<tc> stext[MAXTRAIN];	//存放训练的句子
vector<bool> fltext[MAXTRAIN];	//存放训练句子的标注
vector<bool> ans[MAXTRAIN];	//存放对测试句子标注的结果

class CWS{
	public:
		CWS(){InitFactor();}
		~CWS(){};
		bool Train(char* filename);	//利用文件进行训练
		bool Test(char* filename);	//对文件进行自动分词
		bool Check(char* filename);	//利用文件统计分词结果数据
	private:
		//包装函数：决定分词算法
		bool ifAvg(){return true;}	//包装函数，确认是否需要对参数进行平均化处理
		void TrainAll(){TrainAllS();}	//包装函数，决定用哪种方法进行训练，具体直接调用TrainAllD(),TrainAllP()或TrainAllS()之一实现
		void TestString(vector<tc>& s, vector<bool>& fl, int len){TestStringS(s, fl, len);}	//包装函数，决定用那种方法对长度为len的句子s进行标注，标注存放在fl中，直接调用TestStringD(),TestStringP(),TestStringS()之一实现，应于TrainAll()的选择相同


		//统计量
		int sumtrains;	//训练数据行数
		int sumtest;	//测试数据行数
		int sumtestgram;	//测试数据字数
		int sumunigram;	//训练数据字数
		int sumbigram;	//训练数据bigram个数

		//模式计数
		map<int,int> unigram;	//训练数据每个字出现次数
		map<pii,int> bigram;	//训练数据每2个字出现次数
		map<int,int> cunigram;	//训练数据每个带标记的字出现次数
		map<pbp,int> cbigram;	//训练数据每个pbp(带标注的bigram)出现次数
		map<basic_string<tc>,int> dic;	//训练数据每个词出现次数

		//符号计数
		int st[2];	//标记为0,1出现的次数
		int pst[2][2];	//标记对(0,0)(0,1)(1,0)(1,1)出现次数

		//模式统计
		int numunigram[MAXK + 2];	//出现次数为i的1-gram的种类数, i = 1,2,...,MAXK
		int numbigram[MAXK + 2];	//出现次数为i的2-gram的种类数, i = 1,2,...,MAXK

		//分词算法
		void CountString(vector<tc>& s, vector<bool>& fl, int len);	//对长度为len的训练数据的句子s进行统计，其中标注为fl，统计每个ngram的数量并得到字典
		void InitFactor();	//初始化参数
		void TrainAllP(); //对非结构化感知器进行训练
		void TrainAllS();	//对结构化感知器进行训练
		void TrainAllD(){} 	//正向最大匹配字典法不需要额外训练，空
		void TrainStringP(vector<tc>& s, vector<bool>& fl, int len, double step);//对非结构化感知器利用已标注为fl的长度为len句子s进行训练
		void TrainStringS(vector<tc>& s, vector<bool>& fl, int len, double step);//对结构化感知器利用已标注为fl的长度为len句子s进行训练
		void TrainStringD(){}	//正向最大匹配字典法不需要额外训练，空
		void TestStringP(vector<tc>& s, vector<bool>& fl, int len);	//利用已经训练好的非结构化感知器对长度为len的句子s进行标注，结果存放在fl中
		void TestStringS(vector<tc>& s, vector<bool>& fl, int len);	//利用已经训练好的结构化感知器对长度为len的句子s进行标注，结果存放在fl中
		void TestStringD(vector<tc>& s, vector<bool>& fl, int len);	//利用最大正向匹配对长度为len的句子s进行标注，结果存放在fl中

		//向量处理
		double GetScore(double* a,double* p);	// 利用已知文本特征a和特征向量p计算权值
		void FixFactor(double* a, double* p, double factor);//利用已知文本特征a修正特征向量p，步长为factor

		//概率计算
		void inc(int a);	//给编号为a的1-gram计数加1
		void inc(pii p);	//给编号为 (a,b) 的 2-gram计数+1
		double P(pii gram) { return Katz(gram);}	//计算P(a)
		double P(int gram) { return PGood_Turing(gram);}	//计算P(b|a)
		double PGood_Turing(int gram, int k = MAXK);	//利用Good_Turing计算gram(a)对应的概率P(a)
		double Katz(pii gram, int k = 5, int k2 = MAXK);	//利用Katz回退计算gram(a,b)的概率P(b|a)

		//特征向量
		double pp[MAXFACTOR];	//非结构化感知器特征向量
		double ps[2][MAXFACTOR];//结构化感知器特征向量

		//结果校对
		int tp,fp,fn;	//tp,fp,fn
		double Precision();	//计算Precision
		double Recall();	//计算Recall
		double FScore();	//计算F-score

};

CWS cws;

void
CWS::inc(int a)
{
	if(unigram.find(a) == unigram.end())
	{
		unigram[a] = 1;
		numunigram[1]++;
	}
	else
	{
		int r = unigram[a];
		if(r < MAXK + 2)
			numunigram[r] --;
		r ++;
		unigram[a] = r;
		if(r < MAXK + 2)
			numunigram[r] ++;
	}
}

void
CWS::inc(pii p)
{
	if(bigram.find(p) == bigram.end())
	{
		bigram[p] = 1;
		numbigram[1]++;
	}
	else
	{
		int r = bigram[p];
		if(r < MAXK + 2)
			numbigram[r] --;
		r ++;
		bigram[p] = r;
		if(r < MAXK + 2)
			numbigram[r] ++;
	}
}

#define MAXSFACTOR 14
#define FSET(a) double a[MAXSFACTOR] = {P(lltc),P(lasttc),P(nowtc),P(nexttc),P(nntc),P(pii(lltc,lasttc)),P(pii(lasttc,nowtc)),P(pii(nowtc,nexttc)),P(pii(nexttc,nntc)),P(pii(lltc,nowtc)),P(pii(lasttc,nexttc)),P(pii(nowtc,nntc)),log(cbigram[pbp(false,lasttc,nowtc)]+0.02), log(cbigram[pbp(true,lasttc,nowtc)]+0.02)}


double
CWS::GetScore(double* a, double* p)
{
	double sum = p[0];
	double* b = p;
	for(int i = 0; i < 14; i++)
		sum += a[i]*b[i+1];
	for(int i = 0; i < 12; i ++)
		for(int j = 0; j < 12; j++)
			sum += a[i] * a[j] * b[15 + i*12 + j];
//	for(int i = 0; i < 2; i ++)
//		sum += a[i+12] * b[i+157];
/*
	for(int i = 0; i < 6; i ++)
		for(int j = 0; j < 6; j ++)
			for(int k = 0; k < 6; k ++)
				sum += a[i] * a[j] * a[k] * b[43 + i*36 + j*6 + k];
*/
	return sum;
}

void
CWS::FixFactor(double* a, double* p, double f)
{
	p[0] += 1.0 * f;
	for(int i = 0; i < 14; i++)
		p[i+1] += a[i] * f;
	for(int i = 0; i < 12; i ++)
		for(int j = 0; j < 12; j++)
			p[15 + i*12 + j] += a[i] * a[j] * f;
//	for(int i = 0; i < 2; i ++)
//		p[i+157] += a[i+12] * f;
/*
	for(int i = 0; i < 6; i ++)
		for(int j = 0; j < 6; j ++)
			for(int k = 0; k < 6; k ++)
				p[43 + i*36 + j*6 + k] += a[i] * a[j] * a[k] * f;
*/
}

void
CWS::InitFactor()
{
	pp[0] = 0; ps[0][0] = 0; ps[1][0] = 0;
	for(int i = 1;i<MAXFACTOR;i++){ pp[i] = 0; ps[0][i] = 0; ps[1][i] = 0;}
}
double
CWS::Precision()
{
	return double(tp)/(tp+fp);
}
double
CWS::Recall()
{
	return double(tp)/(tp+fn);
}
double
CWS::FScore()
{
	return 2*Precision()*Recall()/(Precision() + Recall());
}

double
CWS::PGood_Turing(int gram, int k){
	int r = unigram[gram];
	if(r == 0)
		return log(numunigram[1]) - log(sumunigram) - log(sumunigram);
	else if( r <=k )
		return log(r+1) - log(r) + log(numunigram[r+1]) - log(numunigram[r])+ log(r) - log(sumunigram);
	else
		return log(r) - log(sumunigram);
}

double
CWS::Katz(pii gram, int k, int k2){
	double ans;
	if(bigram.find(gram) == bigram.end())
	{
		ans = PGood_Turing(gram.first, k2) + PGood_Turing(gram.second, k2);
	}
	else
	{
		int r = bigram[gram];
		if( r <= k )
		{
			ans = log(r+1) - log(r) + log(numbigram[r+1]) - log(numbigram[r])+ log(r) - log(sumbigram);
		}
		else
			ans = log(r) - log(sumbigram);
	}
	ans -= PGood_Turing(gram.first, k2);
	return ans;
}

bool
CWS::Train(char* filename)
{
	ifstream infile(filename);
	ofstream outfile("outtrain.txt");
	vector<tc> s;
	vector<bool> fl;
	clock_t sTime = clock();

	if(!infile)
	{
		cout<<"TRAIN FILE NOT FOUND"<<endl;
		return false;
	}

	sumtrains = 0;
	sumunigram = 0;
	sumbigram = 0;
	st[0] = 0; st[1] = 0;
	pst[0][0] = 0; pst[0][1] = 0;
	pst[1][0] = 0; pst[1][1] = 0;
	unigram.clear();
	bigram.clear();
	cunigram.clear();
	cbigram.clear();
	dic.clear();

	while(!infile.eof())
	{
		int len = 0;
		int flag = 1;
		basic_string<tc> nows;
		s.clear();
		fl.clear();
		while(!infile.eof())
		{
			char c1,c2,c3;
			c1 = infile.get();
			if(c1 == '\n')break;
			else if(c1 == ' ')
			{
				//dic[nows] ++;
				nows.clear();
				flag = 1;
			}
			else if(c1 < 0 &&((c1 & 0xF0)== 0xE0))
			{
				c2 = infile.get();
				if(c2 == '\n')break;
				else if(c2 >= 0)continue;
				c3 = infile.get();
				if(c3 == '\n')break;
				else if (c3 >= 0)continue;
				s.push_back(tc(c1,c2,c3));
				nows = nows + tc(c1,c2,c3);
				dic[nows] ++;
				fl.push_back(flag);
				flag = 0;
				len++;
			}
			else if (c1 < 0 && (c1 & 0xE0 == 0xC0))
			{
				c2 = infile.get();
				if(c2 == '\n')break;
				else if(c2 >= 0)continue;
				s.push_back(tc(2,c1,c2));
				nows = nows + tc(2,c1,c2);
				dic[nows] ++;
				fl.push_back(flag);
				flag = 0;
				len++;
			}
//			else if(c1 > 0)
//			{
//				s.push_back(tc(1,0,c1));
//				fl.push_back(flag);
//				flag = 0;
//				len++;
//			}
		}
		//dic[nows] ++;
		for(int i = 0; i < len; i++)
		{
			if(fl[i] && i!=0)outfile<<"  ";
			if(s[i].first == 1)
				outfile << s[i].second.second;
			else if (s[i].first == 2)
				outfile << s[i].second.first << s[i].second.second;
			else
				outfile<< s[i].first << s[i].second.first << s[i].second.second;
		}
		outfile<<endl;
		if(infile.eof() || len == 0)break;
		CountString(s, fl, len);
		stext[sumtrains] = s;
		fltext[sumtrains] = fl;
		sumtrains ++;
	}
	InitFactor();
	TrainAll();

	cout<<"Train Done! Time: "<<clock() - sTime<<"ms"<<endl;

	return true;
}

void
CWS::TrainAllP()
{
	unigram[BGRAM] = sumtrains;
	unigram[EGRAM] = sumtrains;

	double spp[MAXFACTOR] = {};
	for(int T = 0; T < MAXSTEPS; T++)
	{
		tp = 0; fp = 0; fn = 0;
		for(int i = 0; i < sumtrains; i++)
		{
			TrainStringP(stext[i],fltext[i],stext[i].size(),T+1);
			for(int j = 0; j< MAXFACTOR; j++) spp[j] += pp[j]/MAXSTEPS/sumtrains;
		}
		cout << "Train Times:" << T <<endl;
		cout << "\tAcc : " << 1 - double(fp + fn)/sumunigram<<endl;
		cout << "\tP : " << Precision() << " R : " << Recall() << " F : " << FScore()<<endl;
//		Test(TESTFILE);
//		Check(ANSWERFILE);
	}
	if(ifAvg())
		for(int i = 0; i < MAXFACTOR ; i++)
			pp[i] = spp[i];
	for(int i = 0; i < MAXFACTOR ; i++)
		cout<< pp[i] <<' ';
	cout<<endl;
}
void
CWS::TrainAllS()
{
	unigram[BGRAM] = sumtrains;
	unigram[EGRAM] = sumtrains;

	double sps[2][MAXFACTOR] = {};
	for(int T = 0; T < MAXSTEPS; T++)
	{
		tp = 0; fp = 0; fn = 0;
		for(int i = 0; i < sumtrains; i++)
		{
			TrainStringS(stext[i],fltext[i],stext[i].size(),(T+1));
			for(int j = 0; j< MAXFACTOR; j++)
			{
				sps[0][j] += ps[0][j]/MAXSTEPS/sumtrains;
				sps[1][j] += ps[1][j]/MAXSTEPS/sumtrains;
			}
		}
		cout << "Train Times:" << T <<endl;
		cout << "\tAcc : " << 1 - double(fp + fn)/sumunigram<<endl;
		cout << "\tP : " << Precision() << " R : " << Recall() << " F : " << FScore()<<endl;
//		Test(TESTFILE);
//		Check(ANSWERFILE);
	}
	if(ifAvg())
	{
		for(int i = 0; i < MAXFACTOR ; i++)
		{
			ps[0][i] = sps[0][i];
			ps[1][i] = sps[1][i];
		}
	}
	for(int i = 0; i < MAXFACTOR; i++)
		cout<< ps[0][i] <<' ';
	cout<<endl;
	for(int i = 0; i < MAXFACTOR; i++)
		cout<< ps[1][i] <<' ';
	cout<<endl;
}


void
CWS::CountString(vector<tc>& s, vector<bool>& f, int len)
{
	int lasttc = BGRAM;
	sumunigram += len;
	sumbigram += (len+1);
	for(int i = 0;i < len; i++)
	{
		int nowtc = tctoi(s[i]);
		inc(nowtc);
		inc(pii(lasttc,nowtc));
		int nowpbi = pbitoi(f[i],nowtc);
		cunigram[nowpbi]++;
		cbigram[pbp(f[i],lasttc,nowtc)]++;
		if(f[i])
		{
			st[1]++;
			if( i == 0 || f[i-1] )
				pst[1][1]++;
			else
				pst[0][1]++;
		}
		else
		{
			st[0]++;
			if(i == 0 || f[i-1] )
				pst[1][0]++;
			else
				pst[0][0]++;
		}
		lasttc = nowtc;
	}
	inc(pii(lasttc,EGRAM));
}

void
CWS::TrainStringP(vector<tc>& s, vector<bool>& fl, int len, double step)
{
	int lltc = BGRAM;
	int lasttc = BGRAM;
	int nowtc = tctoi(s[0]);
	int nexttc;
	int nntc;
	if(len > 1)nexttc = tctoi(s[1]);
	else nexttc = EGRAM;
	if(len > 2)nntc = tctoi(s[2]);
	else nntc = EGRAM;
	for(int k = 0; k < len; k++)
	{
		double *b = pp;
		FSET(a);
//		a[12] = log(cbigram[pbp(false,lasttc,nowtc)]+0.02);
//		a[13] = log(cbigram[pbp(true,lasttc,nowtc)]+0.02);
		//double a[MAXSFACTOR] = {P(lasttc),P(nowtc),P(nexttc),P(pii(lasttc,nowtc)),P(pii(nowtc,nexttc))};
		double sum = GetScore(a,b);
		if((fl[k] && (sum < 0)))
		{
			FixFactor(a, b, 1.0/step);
			fn ++;
		}
		else if((!fl[k] && (sum >= 0)))
		{
			FixFactor(a, b, -1.0/step);
			fp ++;
		}
		else if(fl[k])tp ++;
		lltc = lasttc;
		lasttc = nowtc;
		nowtc = nexttc;
		nexttc = nntc;
		if( k < len - 3)
			nntc = tctoi(s[k+3]);
		else nntc = EGRAM;
	}
}
void
CWS::TrainStringS(vector<tc>& s, vector<bool>& fl, int len, double step)
{
	int lltc = BGRAM;
	int lasttc = BGRAM;
	int nowtc = tctoi(s[0]);
	int nexttc;
	int nntc;
	if(len > 1)nexttc = tctoi(s[1]);
	else nexttc = EGRAM;
	if(len > 2)nntc = tctoi(s[2]);
	else nntc = EGRAM;
	double dp[2][len];
	bool prev[2][len];
	for(int k = 0; k < len; k++)
	{
		FSET(a);
		//double a[MAXSFACTOR] = {P(lasttc),P(nowtc),P(nexttc),P(pii(lasttc,nowtc)),P(pii(nowtc,nexttc))};
		if(k == 0)
		{
			for(int i = 0; i < 2; i++)
			{
				dp[i][k] = log(pst[1][i]) - log(st[1]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]) + GetScore(a,ps[i]);
				prev[i][k] = true;
			}

		}
		else
		{
			for(int i = 0; i < 2; i++)
			{
				double t0 = dp[0][k-1] + log(pst[0][i]) - log(st[0]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]) + GetScore(a,ps[i]);
				double t1 = dp[1][k-1] + log(pst[1][i]) - log(st[1]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]) + GetScore(a,ps[i]);
				if(t0 > t1)
				{
					dp[i][k] = t0;
					prev[i][k] = false;
				}
				else
				{
					dp[i][k] = t1;
					prev[i][k] = true;
				}
			}
		}
		lltc = lasttc;
		lasttc = nowtc;
		nowtc = nexttc;
		nexttc = nntc;
		if( k < len - 3)
			nntc = tctoi(s[k+3]);
		else nntc = EGRAM;
	}

	vector<bool> g;
	bool now;
	if(dp[0][len-1] > dp[1][len-1])now = false;
	else now = true;
	for(int k = len - 1; k >= 0; k--)
	{
		g.push_back(now);
		if(now) now = prev[1][k];
		else now = prev[0][k];
	}
	reverse(g.begin(),g.end());

	lltc = BGRAM;
	lasttc = BGRAM;
	nowtc = tctoi(s[0]);
	if(len > 1)nexttc = tctoi(s[1]);
	else nexttc = EGRAM;
	if(len > 2)nntc = tctoi(s[2]);
	else nntc = EGRAM;
	for(int k = 0; k < len; k++)
	{
		FSET(a);
		//double a[MAXSFACTOR] = {P(lasttc),P(nowtc),P(nexttc),P(pii(lasttc,nowtc)),P(pii(nowtc,nexttc))};
		if((fl[k] && !g[k]))
		{
			FixFactor(a, ps[1], 1.0/step);
			FixFactor(a, ps[0], -1.0/step);
			fn ++;
		}
		else if((!fl[k] && g[k]))
		{
			FixFactor(a, ps[0], 1.0/step);
			FixFactor(a, ps[1], -1.0/step);
			fp ++;
		}
		else if(fl[k])tp ++;
		lltc = lasttc;
		lasttc = nowtc;
		nowtc = nexttc;
		nexttc = nntc;
		if( k < len - 3)
			nntc = tctoi(s[k+3]);
		else nntc = EGRAM;
	}
}

bool
CWS::Test(char* filename)
{
	ifstream infile(filename);
	ofstream outfile("out.txt");
	vector<tc> s;
	vector<bool> fl;
	clock_t sTime = clock();

	if(!infile)
	{
		cout<<"TEST FILE NOT FOUND"<<endl;
		return false;
	}

	sumtest = 0;
	sumtestgram = 0;

	while(!infile.eof())
	{
		int len = 0;
		s.clear();
		fl.clear();
		while(!infile.eof())
		{
			char c1,c2,c3;
			c1 = infile.get();
			if(c1 == '\n')break;
			else if(c1 < 0 &&((c1 & 0xF0)== 0xE0))
			{
				c2 = infile.get();
				if(c2 == '\n')break;
				else if(c2 >= 0)continue;
				c3 = infile.get();
				if(c3 == '\n')break;
				else if (c3 >= 0)continue;
				s.push_back(tc(c1,c2,c3));
				len++;
			}
			else if (c1 < 0 && (c1 & 0xE0 == 0xC0))
			{
				c2 = infile.get();
				if(c2 == '\n')break;
				else if(c2 >= 0)continue;
				s.push_back(tc(2,c1,c2));
				len++;
			}
//			else if(c1 > 0)
//			{
//				s.push_back(tc(1,0,c1));
//				fl.push_back(flag);
//				flag = 0;
//				len++;
//			}
		}
		if(infile.eof() || len == 0)break;
		TestString(s, fl, len);
		for(int i = 0; i < len; i++)
		{
			if(fl[i] && i!=0)outfile<<"  ";
			if(s[i].first == 1)
				outfile << s[i].second.second;
			else if (s[i].first == 2)
				outfile << s[i].second.first << s[i].second.second;
			else
				outfile<< s[i].first << s[i].second.first << s[i].second.second;
		}
		outfile<<endl;
		ans[sumtest] = fl;
		sumtest ++;
		sumtestgram += len;
	}
	cout<<"Test Done! Time: "<<clock() - sTime<<"ms"<<endl;
	return true;
}
void
CWS::TestStringP(vector<tc>& s, vector<bool>& f, int len)
{
	int lltc = BGRAM;
	int lasttc = BGRAM;
	int nowtc = tctoi(s[0]);
	int nexttc;
	int nntc;
	if(len > 1)nexttc = tctoi(s[1]);
	else nexttc = EGRAM;
	if(len > 2)nntc = tctoi(s[2]);
	else nntc = EGRAM;
	for(int k = 0; k < len; k++)
	{
		FSET(a);
//		a[12] = log(cbigram[pbp(false,lasttc,nowtc)]+0.02);
//		a[13] = log(cbigram[pbp(true,lasttc,nowtc)]+0.02);
		//double a[MAXSFACTOR] = {P(lasttc),P(nowtc),P(nexttc),P(pii(lasttc,nowtc)),P(pii(nowtc,nexttc))};
		double sum = GetScore(a, pp);
		if(sum > 0 || k == 0)f.push_back(true);
		else f.push_back(false);
		//if(k == 0 || (log(cbigram[pbp(false,lasttc,nowtc)]+0.02) <= log(cbigram[pbp(true,lasttc,nowtc)]+0.02)))
		//	f.push_back(true);
		//else
		//	f.push_back(false);
		lltc = lasttc;
		lasttc = nowtc;
		nowtc = nexttc;
		nexttc = nntc;
		if( k < len - 3)
			nntc = tctoi(s[k+3]);
		else nntc = EGRAM;
	}
}
void
CWS::TestStringS(vector<tc>& s, vector<bool>& f, int len)
{
	int lltc = BGRAM;
	int lasttc = BGRAM;
	int nowtc = tctoi(s[0]);
	int nexttc;
	int nntc;
	if(len > 1)nexttc = tctoi(s[1]);
	else nexttc = EGRAM;
	if(len > 2)nntc = tctoi(s[2]);
	else nntc = EGRAM;
	double dp[2][len];
	bool prev[2][len];
	for(int k = 0; k < len; k++)
	{
		FSET(a);
		//double a[MAXSFACTOR] = {P(lasttc),P(nowtc),P(nexttc),P(pii(lasttc,nowtc)),P(pii(nowtc,nexttc))};
		if(k == 0)
		{
			for(int i = 0; i < 2; i++)
			{
				dp[i][k] = log(pst[1][i]) - log(st[1]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]) + GetScore(a,ps[i]);
				//dp[i][k] = log(pst[1][i]) - log(st[1]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]);
				prev[i][k] = true;
			}

		}
		else
		{
			for(int i = 0; i < 2; i++)
			{
				double t0 = dp[0][k-1] + log(pst[0][i]) - log(st[0]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]) + GetScore(a,ps[i]);
				double t1 = dp[1][k-1] + log(pst[1][i]) - log(st[1]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]) + GetScore(a,ps[i]);
				//double t0 = dp[0][k-1] + log(pst[0][i]) - log(st[0]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]);
				//double t1 = dp[1][k-1] + log(pst[1][i]) - log(st[1]) + log(cbigram[pbp(i,lasttc,nowtc)]+0.02) - log(st[i]);
				if(t0 > t1)
				{
					dp[i][k] = t0;
					prev[i][k] = false;
				}
				else
				{
					dp[i][k] = t1;
					prev[i][k] = true;
				}
			}
		}
		lltc = lasttc;
		lasttc = nowtc;
		nowtc = nexttc;
		nexttc = nntc;
		if( k < len - 3)
			nntc = tctoi(s[k+3]);
		else nntc = EGRAM;
	}
	f.clear();
	bool now;
	if(dp[0][len-1] > dp[1][len-1])now = false;
	else now = true;
	for(int k = len - 1; k >= 0; k--)
	{
		f.push_back(now);
		if(now) now = prev[1][k];
		else now = prev[0][k];
	}
	reverse(f.begin(),f.end());

}
void
CWS::TestStringD(vector<tc>& s, vector<bool>& f, int len)
{
	int now = 0;
	basic_string<tc> nows;
	while( now < len)
	{
		f.push_back(true);
		nows.clear();
		int next = now;
		for(int k = now; k < len; k ++)
		{
			nows = nows + s[k];
			if((dic.find(nows) != dic.end())&&(dic[nows] >= MINWORD))
				next = k;
			else
				break;
		}
		for(int k = now + 1; k <= next; k++)
			f.push_back(false);
		now = next + 1;
	}
}

bool
CWS::Check(char* filename)
{
	ifstream infile(filename);
	ofstream outfile("check.txt");
	vector<tc> s;
	vector<bool> fl;
	clock_t sTime = clock();

	if(!infile)
	{
		cout<<"ANSWER FILE NOT FOUND"<<endl;
		return false;
	}

	int now = 0;
	sumtestgram = 0;
	tp = 0;
	fp = 0;
	fn = 0;

	while(!infile.eof())
	{
		int len = 0;
		int flag = 1;
		s.clear();
		fl.clear();
		while(!infile.eof())
		{
			char c1,c2,c3;
			c1 = infile.get();
			if(c1 == '\n')break;
			else if(c1 == ' ') flag = 1;
			else if(c1 < 0 &&((c1 & 0xF0)== 0xE0))
			{
				c2 = infile.get();
				if(c2 == '\n')break;
				else if(c2 >= 0)continue;
				c3 = infile.get();
				if(c3 == '\n')break;
				else if (c3 >= 0)continue;
				s.push_back(tc(c1,c2,c3));
				fl.push_back(flag);
				flag = 0;
				len++;
			}
			else if (c1 < 0 && (c1 & 0xE0 == 0xC0))
			{
				c2 = infile.get();
				if(c2 == '\n')break;
				else if(c2 >= 0)continue;
				s.push_back(tc(2,c1,c2));
				fl.push_back(flag);
				flag = 0;
				len++;
			}
//			else if(c1 > 0)
//			{
//				s.push_back(tc(1,0,c1));
//				fl.push_back(flag);
//				flag = 0;
//				len++;
//			}
		}
		if(infile.eof() || len == 0)break;
		for(int i = 0; i < len; i++)
		{
//			if(i == 0)ans[now].push_back(true);
//			else ans[now].push_back(false);
//			ans[now].push_back(true);
			if(fl[i] && ans[now][i])tp++;
			else if(fl[i] && !ans[now][i])fn++;
			else if(!fl[i] && ans[now][i])fp++;
		}
		now++;
		sumtestgram += len;
	}
	cout<<"Check Done! Time: "<<clock() - sTime<<"ms"<<endl;
	cout << "Acc : " << 1 - double(fp + fn)/sumtestgram<<endl;
	cout << "P : " << Precision() << " R : " << Recall() << " F : " << FScore()<<endl;
	return true;
}

int main()
{
	cws.Train(TRAINFILE);
	cws.Test(TESTFILE);
	cws.Check(ANSWERFILE);
	return 0;
}
